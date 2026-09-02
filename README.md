# Controlled real-geometry FiberTrace validation

This CPU-only experiment rasterizes two real WebKnossos NML fiber paths into
native-compatible `presence` / `nx` / `ny` fields. It is a controlled search
experiment, not a neural-model evaluation and not a standalone tracer

The field frame is local XYZ: absolute NML XYZ minus the recorded ROI origin.
Arrays are stored as ZYX uint8 Zarr v2, and their Lasagna manifests declare
source-to-base scale 1. The native harness therefore uses inference scaledown
power 0. Normals are deliberately artificial: constant +Z (`grad_mag=255`,
`nx=ny=128`) so normal-aware terms remain active but invariant

## Local preparation

On the constrained Intel Mac, field generation and analysis utilities work;
the pinned native C++ build does not. Run from this directory:

```sh
python3 -m unittest discover -s python/tests -v
python3 python/generate_controlled_fields.py \
  --nml ../villa-upstream-main/foundation/datasets/fibers-dataset/fibers_s5_06500z_02000y_04000x_500_v03.nml \
  --cases cases/smoke_spans.json --output generated
```

## Linux native validation

The recommended native baseline is the included Ubuntu 24.04 GitHub Actions
workflow, using Clang as in current upstream VC CI. It checks out villa at
`908aa7f06e326d6df5cf167ebaab1fc733466987`, generates the fields, builds only
the injected non-GUI target, and runs the frozen four cases. To reproduce that
run on Ubuntu 24.04 after checking out this project and pinned villa side by
side, install the CI packages and run:

```sh
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential clang cmake ninja-build git pkg-config \
  libblosc-dev libboost-dev libceres-dev libcurl4-openssl-dev \
  libeigen3-dev liblz4-dev libopencv-dev libsuitesparse-dev \
  libtiff-dev libzstd-dev nlohmann-json3-dev zlib1g-dev
python3 -m pip install 'numpy>=1.24'

cmake -S ../villa-upstream-main/volume-cartographer -B build-vc-native \
  -G Ninja -DCMAKE_BUILD_TYPE=QuickBuild \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DVC_BUILD_APPS=OFF -DVC_BUILD_UI_TRACER=OFF \
  -DVC_BUILD_FLATBOI=OFF -DVC_BUILD_PYTHON=OFF -DVC_TESTING=OFF \
  -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES="$PWD/cmake/InjectNativeHarness.cmake" \
  -DFIBER_TRACE_CONTROLLED_VALIDATION_ROOT="$PWD"
cmake --build build-vc-native --target native_trace_segment -j2

build-vc-native/bin/native_trace_segment \
  --spans generated/spans_local.json --output generated/native_results.json \
  --inference-scaledown-power 0
python3 python/analyze_results.py --spans generated/spans_local.json \
  --results generated/native_results.json --output generated/reference_distances.json
```

`generated/` is reproducible and disposable. It contains no CT, neural model,
or copied upstream source

The fields preserve real human NML geometry, but their prediction and normal
values are controlled rather than neural predictions. This isolates native
tracing/search behavior; it makes no claim about neural prediction quality
