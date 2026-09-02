#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"

#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
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
    bool selfTestFilter = false;
};

enum class PredictionSourceMode {
    Native,
    Delegated,
    ZeroInvalid,
};

constexpr std::array<PredictionSourceMode, 3> kPredictionSourceModes{
    PredictionSourceMode::Native,
    PredictionSourceMode::Delegated,
    PredictionSourceMode::ZeroInvalid,
};

[[nodiscard]] const char* predictionSourceModeName(PredictionSourceMode mode)
{
    switch (mode) {
    case PredictionSourceMode::Native:
        return "native";
    case PredictionSourceMode::Delegated:
        return "delegated";
    case PredictionSourceMode::ZeroInvalid:
        return "zero_invalid";
    }
    throw std::logic_error("unknown prediction source mode");
}

[[nodiscard]] vc::fiber_tracer::FiberPredictionSample transformSample(
    vc::fiber_tracer::FiberPredictionSample sample,
    bool invalidateZeroPresence)
{
    if (!invalidateZeroPresence)
        return sample;

    vc::fiber_tracer::FiberPredictionSampleOptions transformed;
    transformed.reserve(sample.options.size());
    for (const auto& option : sample.options) {
        auto transformedOption = option;
        if (transformedOption.valid && transformedOption.presence == 0.0f)
            transformedOption.valid = false;
        transformed.push_back(transformedOption);
    }
    sample.options = std::move(transformed);
    return sample;
}

void transformSamples(
    std::vector<vc::fiber_tracer::FiberPredictionSample>& samples,
    bool invalidateZeroPresence)
{
    if (!invalidateZeroPresence)
        return;
    for (auto& sample : samples)
        sample = transformSample(std::move(sample), true);
}

class DelegatingPredictionSource final : public vc::fiber_tracer::FiberPredictionSource {
public:
    DelegatingPredictionSource(
        const vc::fiber_tracer::FiberPredictionField& field,
        bool invalidateZeroPresence)
        : field_(field)
        , invalidateZeroPresence_(invalidateZeroPresence)
    {
    }

    [[nodiscard]] bool supportsConcurrentSampling() const noexcept override
    {
        return field_.supportsConcurrentSampling();
    }

    [[nodiscard]] vc::lasagna::NormalPrefetchReport prefetchSamples(
        const std::vector<cv::Vec3d>& volumePoints) const override
    {
        return field_.prefetchSamples(volumePoints);
    }

    void sampleBatch(
        const std::vector<cv::Vec3d>& volumePoints,
        const std::vector<cv::Vec3d>& referenceDirections,
        int parallelThreads,
        std::vector<vc::fiber_tracer::FiberPredictionSample>& samples) const override
    {
        field_.sampleBatch(volumePoints, referenceDirections, parallelThreads, samples);
        transformSamples(samples, invalidateZeroPresence_);
    }

    [[nodiscard]] vc::fiber_tracer::FiberPredictionSample sample(
        const cv::Vec3d& volumePoint,
        const cv::Vec3d& referenceDirection) const override
    {
        return transformSample(
            field_.sample(volumePoint, referenceDirection), invalidateZeroPresence_);
    }

private:
    const vc::fiber_tracer::FiberPredictionField& field_;
    bool invalidateZeroPresence_;
};

[[nodiscard]] bool sameSample(
    const vc::fiber_tracer::FiberPredictionSample& left,
    const vc::fiber_tracer::FiberPredictionSample& right)
{
    if (left.options.size() != right.options.size())
        return false;
    for (size_t index = 0; index < left.options.size(); ++index) {
        const auto& lhs = left.options[index];
        const auto& rhs = right.options[index];
        if (lhs.presence != rhs.presence || lhs.valid != rhs.valid ||
            lhs.direction[0] != rhs.direction[0] || lhs.direction[1] != rhs.direction[1] ||
            lhs.direction[2] != rhs.direction[2]) {
            return false;
        }
    }
    return true;
}

void requireFilterTest(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(std::string("prediction-source filter self-test failed: ") + message);
}

void runPredictionSourceFilterSelfTest()
{
    using vc::fiber_tracer::FiberPredictionSample;

    FiberPredictionSample source;
    source.options.push_back({cv::Vec3f{1.0f, 2.0f, 3.0f}, 0.75f, true});
    source.options.push_back({cv::Vec3f{4.0f, 5.0f, 6.0f}, 0.0f, true});
    source.options.push_back({cv::Vec3f{7.0f, 8.0f, 9.0f}, 0.0f, false});
    source.options.push_back({cv::Vec3f{10.0f, 11.0f, 12.0f}, 0.25f, false});

    const auto passThrough = transformSample(source, false);
    requireFilterTest(sameSample(source, passThrough), "pass-through changed an option");

    const auto filtered = transformSample(source, true);
    requireFilterTest(filtered.options.size() == source.options.size(), "option count changed");
    requireFilterTest(filtered.options[0].valid, "positive presence valid option was invalidated");
    requireFilterTest(!filtered.options[1].valid, "zero-presence valid option remained valid");
    requireFilterTest(!filtered.options[2].valid, "already-invalid zero-presence option changed");
    requireFilterTest(!filtered.options[3].valid, "already-invalid positive-presence option changed");
    for (size_t index = 0; index < source.options.size(); ++index) {
        requireFilterTest(
            filtered.options[index].presence == source.options[index].presence &&
                filtered.options[index].direction[0] == source.options[index].direction[0] &&
                filtered.options[index].direction[1] == source.options[index].direction[1] &&
                filtered.options[index].direction[2] == source.options[index].direction[2],
            "filter changed presence or direction");
    }

    std::vector<FiberPredictionSample> batch{source, source};
    transformSamples(batch, true);
    requireFilterTest(
        sameSample(filtered, batch[0]) && sameSample(filtered, batch[1]),
        "batch filtering differs from scalar filtering");
    std::cout << "prediction-source filter self-test passed\n";
}

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
        } else if (argument == "--self-test-filter") {
            options.selfTestFilter = true;
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage: native_trace_segment --spans PATH --output PATH "
                << "[--fiber-manifest PATH] [--normal-manifest PATH] "
                << "[--inference-scaledown-power 0]\n"
                << "       native_trace_segment --self-test-filter\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.selfTestFilter && (!options.spansPath.empty() || !options.outputPath.empty()))
        throw std::invalid_argument("--self-test-filter cannot be combined with tracing arguments");
    if (!options.selfTestFilter && (options.spansPath.empty() || options.outputPath.empty()))
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
    const std::filesystem::path& spansDirectory,
    PredictionSourceMode sourceMode)
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

    const DelegatingPredictionSource delegatedSource(
        predictionField, sourceMode == PredictionSourceMode::ZeroInvalid);
    const vc::fiber_tracer::FiberPredictionSource& predictions =
        sourceMode == PredictionSourceMode::Native
            ? static_cast<const vc::fiber_tracer::FiberPredictionSource&>(predictionField)
            : static_cast<const vc::fiber_tracer::FiberPredictionSource&>(delegatedSource);

    const auto started = Clock::now();
    const auto result = vc::fiber_tracer::traceFiberSegment(
        predictions, request, &normalSampler);
    const double elapsed = std::chrono::duration<double>(Clock::now() - started).count();

    Json output{
        {"case_id", span.at("id")},
        {"prediction_source_mode", predictionSourceModeName(sourceMode)},
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
        {"forward_reached_trace_length", result.forward.reachedTraceLength},
        {"reverse_reached_trace_length", result.reverse.reachedTraceLength},
        {"forward_steps", result.forward.steps},
        {"reverse_steps", result.reverse.steps},
        {"forward_endpoint_error_trace_voxels", result.forwardEndpointErrorTraceVoxels},
        {"reverse_endpoint_error_trace_voxels", result.reverseEndpointErrorTraceVoxels},
        {"max_endpoint_error_trace_voxels", result.maxEndpointErrorTraceVoxels},
        {"max_endpoint_error_base_voxels", result.maxEndpointErrorBaseVoxels},
        {"meeting_error_trace_voxels", result.meetingErrorTraceVoxels},
        {"meeting_error_base_voxels", result.meetingErrorBaseVoxels},
        {"meeting_error_ratio", result.meetingErrorRatio},
        {"meeting_trace_length_trace_voxels", result.meetingTraceLengthTraceVoxels},
        {"fused_path_length_trace_voxels", polylineLength(result.fusedLine)},
        {"elapsed_wall_seconds", elapsed},
        {"fused_path_xyz", writePath(result.fusedLine)},
        {"forward_points_xyz", writePath(result.forward.points)},
        {"reverse_points_xyz", writePath(result.reverse.points)},
        {"trace_config", traceConfigJson(request.config, options.inferenceScaledownPower)},
    };
    if (span.contains("competing_tree_id"))
        output["competing_tree_id"] = span.at("competing_tree_id");
    return output;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const Options options = parseArgs(argc, argv);
        if (options.selfTestFilter) {
            runPredictionSourceFilterSelfTest();
            return 0;
        }
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
        for (const auto& span : spans.at("cases")) {
            for (const auto sourceMode : kPredictionSourceModes) {
                output["results"].push_back(
                    runSpan(span, options, spansDirectory, sourceMode));
            }
        }

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
