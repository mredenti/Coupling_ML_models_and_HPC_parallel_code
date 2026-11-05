# Building and testing

Run these commands from the repository root. CMake 3.20 or newer, a C++20
compiler, and MPI are required. OpenMP is enabled by default.

Presets do not load modules or set compiler, MPI, or library search paths.
Prepare the environment yourself before configuring, building, or debugging.
CMake discovers the active toolchain; a new build directory also honors `CXX`
if you set it. Existing build directories retain cached toolchain paths, so
use a fresh directory when switching compilers or MPI installations.

## Leonardo environment

```bash
module purge
module load gcc/12.2.0 openmpi/4.1.6--gcc--12.2.0-cuda-12.2
```

## Configuration presets

One tracked `CMakePresets.json` contains all shared configurations:

| Configure/build preset | Coupling | Optional ONNX engine | Tests | Build type |
| --- | --- | --- | --- | --- |
| `case-a` | A | No | Yes | Debug |
| `case-b` | B | No | Yes | Debug |
| `case-a-onnx` | A | Yes | Yes | Debug |
| `case-b-onnx` | B | Yes | Yes | Debug |
| `case-a-onnx-debug` | A | Yes | Yes | Debug (`-O0 -g`) |
| `case-a-no-tests` | A | No | No | RelWithDebInfo |
| `case-b-no-tests` | B | No | No | RelWithDebInfo |
| `case-a-onnx-no-tests` | A | Yes | No | RelWithDebInfo |
| `case-b-onnx-no-tests` | B | Yes | No | RelWithDebInfo |

Every preset has its own `build/<preset-name>/` directory.

The dedicated-node experiment called "C" uses the Case B executable with
`--ml-server-nodes=1`. It is a runtime topology, not a third compile-time case.

`MLHPC_BACKEND=DUMMY|ONNX` selects the implementation compiled into the backend
factory. The ordinary presets select `DUMMY`; the `-onnx` presets select `ONNX`.
The factory target links the core library and, for ONNX builds only, the runtime
adapter. The application links the factory target, inheriting these dependencies.

List the available presets:

```bash
cmake --list-presets
cmake --build --list-presets
ctest --list-presets
```

- Case A means local inference on every rank. 
- Case B means MPI client/server inference; the preset does not fix server placement. At runtime,
`--ml-server-nodes=0` selects one server rank per node, while a positive valid
count reserves whole inference nodes.

## Example: Build and test Case A

```bash
cmake --preset case-a
cmake --build --preset case-a
ctest --preset case-a-unit
```

Run the MPI tests in a one-node allocation with capacity for four MPI ranks:

```bash
ctest --preset case-a-mpi
```

The distributor test creates its fixture under `/tmp`, so this MPI test group
should be run on one node. It is not a multi-node scaling benchmark.

For Case B:

```bash
cmake --preset case-b
cmake --build --preset case-b
ctest --preset case-b-unit
ctest --preset case-b-mpi
```

The coupling tests construct the selected backend through the same factory as
the application, while a separate native `DummyBackend` supplies reference
values. Case A verifies that every rank owns a backend and rejects a missing
one. Case B verifies that only server ranks own backends and sends chunked
particle batches from clients with no local backend.

The MPI group also includes `test_backend_smoke`, which checks application
startup, backend creation, and shutdown without reading a snapshot.

## Build and test the ONNX engine

Set the serving-model location once per shell or batch job:

```bash
export MLHPC_ONNX_MODEL="$PWD/models/dummy.onnx"
```

This test requires the two-input dummy-equivalent model exported by `python/create_dummy_onnx.py`,
not the three-input cooling MLP. The exported model reproduces `0.5*rho + 0.25*u` for a controlled numerical
comparison, not a physical cooling prediction. The presets download the ONNX Runtime library, not the model.

```bash
cmake --preset case-a-onnx
cmake --build --preset case-a-onnx
ctest --preset case-a-onnx
```

To see successful test output:

```bash
ctest --preset case-a-onnx --verbose
```

The comparison test should report:

```text
ONNX backend matches DummyBackend; batch sizes 0, 1, 4, 17 passed
```

The corresponding Case B commands use `case-b-onnx`.

To test selected-backend inference through coupling, use a one-node allocation
with capacity for four MPI ranks:

```bash
ctest --preset case-a-onnx-mpi --verbose
```

For the local-server topology:

```bash
cmake --preset case-b-onnx
cmake --build --preset case-b-onnx
ctest --preset case-b-onnx --verbose
ctest --preset case-b-onnx-mpi --verbose
```

## Run the selected backend

For an ONNX build, pass `--model=PATH` at runtime, even when running only the
topology smoke test. `MLHPC_ONNX_MODEL` configures test arguments; it does not
embed a model path in the executable.

Within an allocation:

```bash
mpirun --np 4 build/case-a-onnx/mlhpc --model="$MLHPC_ONNX_MODEL"
mpirun --np 4 build/case-b-onnx/mlhpc --model="$MLHPC_ONNX_MODEL"
```

Append a compatible gas snapshot path for the full read/convert/evaluate
pipeline. For the dedicated-node layout, use the B executable on at least two
nodes and add `--ml-server-nodes=1`. Only ranks on the selected server node
create engines. The same `rho`, `u`, and optional particle fields travel over
the unchanged MPI wire format.

## Reuse a runtime without downloading

On a login node, the first ONNX configuration can download the runtime.
For subsequent configurations or compute-node jobs without internet access:

```bash
cmake --preset case-b-onnx \
    -DMLHPC_DOWNLOAD_ONNX=OFF \
    -DMLHPC_ONNX_ROOT="$PWD/build/case-a-onnx/_deps/onnxruntime-src"
```

Keep the referenced runtime directory available while executables use it!
Reapply these overrides when configuring that preset again, since shared
presets intentionally specify their default download settings.
