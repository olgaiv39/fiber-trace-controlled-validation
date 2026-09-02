#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

struct Options {
    std::string fiberManifest;
    std::string normalManifest;
    std::filesystem::path spansPath;
    std::filesystem::path outputPath;
    int inferenceScaledownPower = 0;
};

[[nodiscard]] std::string requireValue(int& index, int argc, char** argv, const char* name)
{
    if (++index >= argc)
        throw std::invalid_argument(std::string("missing value for ") + name);
    return argv[index];
}

[[nodiscard]] Options parseArgs(int argc, char** argv)
{
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--fiber-manifest") {
            options.fiberManifest = requireValue(index, argc, argv, "--fiber-manifest");
        } else if (argument == "--normal-manifest") {
            options.normalManifest = requireValue(index, argc, argv, "--normal-manifest");
        } else if (argument == "--spans") {
            options.spansPath = requireValue(index, argc, argv, "--spans");
        } else if (argument == "--output") {
            options.outputPath = requireValue(index, argc, argv, "--output");
        } else if (argument == "--inference-scaledown-power") {
            options.inferenceScaledownPower = std::stoi(
                requireValue(index, argc, argv, "--inference-scaledown-power"));
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage: native_trace_segment --spans PATH --output PATH "
                << "[--fiber-manifest PATH] [--normal-manifest PATH] "
                << "[--inference-scaledown-power 0]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.spansPath.empty() || options.outputPath.empty())
        throw std::invalid_argument("--spans and --output are required");
    return options;
}

[[nodiscard]] cv::Vec3d readPoint(const Json& point)
{
    if (!point.is_array() || point.size() != 3)
        throw std::runtime_error("point must be an XYZ array of length three");
    return {point.at(0).get<double>(), point.at(1).get<double>(), point.at(2).get<double>()};
}

[[nodiscard]] std::vector<cv::Vec3d> readReferenceLine(const Json& span)
{
    if (!span.contains("reference_line_xyz_local") ||
        !span.at("reference_line_xyz_local").is_array() ||
        span.at("reference_line_xyz_local").size() < 2) {
        throw std::runtime_error(
            "span must contain a reference_line_xyz_local array with at least two points");
    }
    std::vector<cv::Vec3d> line;
    line.reserve(span.at("reference_line_xyz_local").size());
    for (const auto& point : span.at("reference_line_xyz_local"))
        line.push_back(readPoint(point));
    return line;
}

[[nodiscard]] size_t readReferenceIndex(
    const Json& span,
    const char* key,
    size_t lineSize)
{
    if (!span.contains(key) ||
        (!span.at(key).is_number_integer() && !span.at(key).is_number_unsigned())) {
        throw std::runtime_error(std::string("span must contain integer ") + key);
    }
    const auto value = span.at(key).get<long long>();
    if (value < 0 || static_cast<size_t>(value) >= lineSize)
        throw std::runtime_error(std::string("span ") + key + " is out of range");
    return static_cast<size_t>(value);
}

[[nodiscard]] Json writePoint(const cv::Vec3d& point)
{
    return Json::array({point[0], point[1], point[2]});
}

[[nodiscard]] Json writePath(const std::vector<cv::Vec3d>& path)
{
    Json out = Json::array();
    for (const auto& point : path)
        out.push_back(writePoint(point));
    return out;
}

[[nodiscard]] double polylineLength(const std::vector<cv::Vec3d>& path)
{
    double length = 0.0;
    for (size_t index = 1; index < path.size(); ++index)
        length += cv::norm(path[index] - path[index - 1]);
    return length;
}

[[nodiscard]] std::string spanManifest(
    const Json& span,
    const char* key,
    const std::string& fallback)
{
    if (!fallback.empty())
        return fallback;
    if (!span.contains(key) || !span.at(key).is_string())
        throw std::runtime_error(std::string("span is missing ") + key);
    return span.at(key).get<std::string>();
}

[[nodiscard]] std::string resolveManifestPath(
    const std::string& manifest,
    const std::filesystem::path& spansDirectory)
{
    const std::filesystem::path path(manifest);
    return (path.is_absolute() ? path : spansDirectory / path).string();
}

[[nodiscard]] Json traceConfigJson(
    const vc::fiber_tracer::FiberTraceConfig& config,
    int inferenceScaledownPower)
{
    return {
        {"step_voxels", config.stepVoxels},
        {"cone_angle_degrees", config.coneAngleDegrees},
        {"cone_angle_step_degrees", config.coneAngleStepDegrees},
        {"beam_width", config.beamWidth},
        {"max_step_factor", config.maxStepFactor},
        {"meeting_accept_max_error_ratio", config.meetingAcceptMaxErrorRatio},
        {"endpoint_accept_threshold_base_voxels", config.endpointAcceptThresholdBaseVoxels},
        {"smoothness_weight", config.smoothnessWeight},
        {"smoothness_normal_weight", config.smoothnessNormalWeight},
        {"smoothness_tangent_weight", config.smoothnessTangentWeight},
        {"cumulative_smoothness_tangent_weight", config.cumulativeSmoothnessTangentWeight},
        {"parallel_threads", config.parallelThreads},
        {"trace_to_base_scale", config.traceToBaseScale},
        {"inference_scaledown_power", inferenceScaledownPower},
    };
}

[[nodiscard]] Json runSpan(
    const Json& span,
    const Options& options,
    const std::filesystem::path& spansDirectory)
{
    const std::string fiberManifest = resolveManifestPath(
        spanManifest(span, "fiber_manifest", options.fiberManifest), spansDirectory);
    const std::string normalManifest = resolveManifestPath(
        spanManifest(span, "normal_manifest", options.normalManifest), spansDirectory);
    const auto openedField = vc::lasagna::LasagnaDataset::openLocation(fiberManifest);
    const auto scales = vc::fiber_tracer::resolveFiberPredictionTraceScales(
        openedField.manifest(), options.inferenceScaledownPower);
    auto fieldManifest = openedField.manifest();
    fieldManifest.workingToBaseScale = scales.traceToBaseScale;
    const vc::lasagna::LasagnaDataset fieldDataset(std::move(fieldManifest));
    const vc::fiber_tracer::FiberPredictionField predictionField(fieldDataset, 128ULL * 1024ULL * 1024ULL);

    vc::lasagna::LasagnaDatasetOpenOptions normalOptions;
    normalOptions.workingToBaseScale = scales.traceToBaseScale;
    const auto normalDataset = vc::lasagna::LasagnaDataset::openLocation(normalManifest, normalOptions);
    const vc::lasagna::LasagnaNormalSampler normalSampler(
        normalDataset, vc::lasagna::LasagnaNormalSamplerOptions{128ULL * 1024ULL * 1024ULL});

    if (!span.contains("endpoint_xyz_local") || span.at("endpoint_xyz_local").size() != 2)
        throw std::runtime_error("span must contain exactly two endpoint_xyz_local points");
    const cv::Vec3d startEndpoint = readPoint(span.at("endpoint_xyz_local").at(0));
    const cv::Vec3d targetEndpoint = readPoint(span.at("endpoint_xyz_local").at(1));
    std::vector<cv::Vec3d> referenceLine = readReferenceLine(span);
    const size_t startIndex = readReferenceIndex(
        span, "start_index", referenceLine.size());
    const size_t targetIndex = readReferenceIndex(
        span, "target_index", referenceLine.size());
    constexpr double kEndpointTolerance = 1.0e-9;
    if (cv::norm(referenceLine[startIndex] - startEndpoint) > kEndpointTolerance ||
        cv::norm(referenceLine[targetIndex] - targetEndpoint) > kEndpointTolerance) {
        throw std::runtime_error(
            "span reference-line indices do not match endpoint_xyz_local coordinates");
    }
    vc::fiber_tracer::FiberTraceSegmentRequest request;
    request.referenceLine = std::move(referenceLine);
    request.startIndex = startIndex;
    request.targetIndex = targetIndex;
    request.config.traceToBaseScale = scales.traceToBaseScale;
    request.config.baseVoxelSizeUm = 7.91;
    request.config.parallelThreads = 1;

    const auto started = Clock::now();
    const auto result = vc::fiber_tracer::traceFiberSegment(
        predictionField, request, &normalSampler);
    const double elapsed = std::chrono::duration<double>(Clock::now() - started).count();

    return {
        {"case_id", span.at("id")},
        {"intended_tree_id", span.at("intended_tree_id")},
        {"endpoint_node_ids", span.at("endpoint_node_ids")},
        {"fiber_manifest", fiberManifest},
        {"normal_manifest", normalManifest},
        {"accepted", result.accepted},
        {"reason", result.reason},
        {"detail", result.detail},
        {"forward_reason", result.forward.reason},
        {"reverse_reason", result.reverse.reason},
        {"forward_reached_target_plane", result.forward.reachedTargetPlane},
        {"reverse_reached_target_plane", result.reverse.reachedTargetPlane},
        {"meeting_error_trace_voxels", result.meetingErrorTraceVoxels},
        {"meeting_error_base_voxels", result.meetingErrorBaseVoxels},
        {"meeting_error_ratio", result.meetingErrorRatio},
        {"meeting_trace_length_trace_voxels", result.meetingTraceLengthTraceVoxels},
        {"fused_path_length_trace_voxels", polylineLength(result.fusedLine)},
        {"elapsed_wall_seconds", elapsed},
        {"fused_path_xyz", writePath(result.fusedLine)},
        {"trace_config", traceConfigJson(request.config, options.inferenceScaledownPower)},
    };
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const Options options = parseArgs(argc, argv);
        std::ifstream input(options.spansPath);
        if (!input)
            throw std::runtime_error("cannot open spans file: " + options.spansPath.string());
        Json spans;
        input >> spans;
        if (!spans.contains("cases") || !spans.at("cases").is_array())
            throw std::runtime_error("spans file must contain a cases array");

        Json output;
        output["experiment"] = "controlled_real_nml_geometry_not_neural_inference";
        output["villa_commit"] = spans.value("villa_commit", "unknown");
        output["spans_path"] = options.spansPath.string();
        output["results"] = Json::array();
        const auto spansDirectory = options.spansPath.parent_path();
        for (const auto& span : spans.at("cases"))
            output["results"].push_back(runSpan(span, options, spansDirectory));

        std::ofstream destination(options.outputPath);
        if (!destination)
            throw std::runtime_error("cannot write output: " + options.outputPath.string());
        destination << output.dump(2) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "native_trace_segment: " << error.what() << '\n';
        return 1;
    }
}
