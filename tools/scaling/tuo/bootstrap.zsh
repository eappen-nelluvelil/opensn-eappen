#!/bin/zsh

set -euo pipefail

script=${0:A}
framework_source=${script:h:h:h:h}
source_dir=${OPENSN_SOURCE:-${script:h:h:h:h}}
work_root=${OPENSN_TUO_ROOT:-/usr/workspace/${USER}/opensn-gpu/tuo/gfx942}
gmsh_version=${OPENSN_TUO_GMSH_VERSION:-4.15.2}
stack_id=${OPENSN_TUO_STACK_ID:-}

typeset -a python_packages=(
  pip==26.1.1
  setuptools==82.0.1
  wheel==0.47.0
  pybind11==3.0.4
  numpy==2.4.4
  jinja2==3.1.6
  matplotlib==3.10.9
  ninja==1.13.0
  gmsh==$gmsh_version
  MarkupSafe==3.0.3
  contourpy==1.3.3
  cycler==0.12.1
  fonttools==4.62.1
  kiwisolver==1.5.0
  packaging==26.3
  pillow==12.2.0
  pyparsing==3.3.2
  python-dateutil==2.9.0.post0
  six==1.17.0
)
mpi4py_package=mpi4py==4.1.1

usage()
{
  print -u2 \
    "usage: $0 {configure-deps|build-deps|build-opensn|verify-build|paths|all}"
  exit 2
}

initialize_modules()
{
  (( $+functions[module] || $+commands[module] )) && return
  local candidate
  for candidate in \
    ${MODULESHOME:-}/init/zsh \
    /usr/share/lmod/lmod/init/zsh \
    /etc/profile.d/z00_lmod.sh
  do
    if [[ -n $candidate && -r $candidate ]]; then
      source "$candidate"
      return
    fi
  done
  print -u2 'Unable to initialize the module command.'
  exit 1
}

load_toolchain()
{
  initialize_modules
  unset CMAKE_PREFIX_PATH PETSC_DIR PETSC_ARCH HDF5_ROOT HDF5_DIR VTK_DIR
  unset caliper_DIR Boost_DIR Boost_ROOT MPICPP_LITE_DIR PKG_CONFIG_PATH
  unset CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH LIBRARY_PATH PYTHONPATH PYTHONHOME
  unset VIRTUAL_ENV
  module purge
  module load \
    python/3.13.2 \
    cmake/3.29.2 \
    rocm/7.2.1 \
    rocmcc/7.2.1-magic \
    cray-mpich/9.1.0
  export CC=amdclang
  export CXX=amdclang++
  export MPICH_GPU_SUPPORT_ENABLED=1
  export MPICH_SMP_SINGLE_COPY_MODE=XPMEM
}

source_revision()
{
  git -C "$source_dir" rev-parse --verify HEAD^{commit}
}

require_clean_source()
{
  git -C "$source_dir" rev-parse --git-dir >/dev/null
  local revision=$(source_revision)
  [[ ${#revision} == 40 && $revision != *[^0-9a-f]* ]] || {
    print -u2 "Unable to resolve an exact source revision in $source_dir"
    exit 1
  }
  local status=$(git -C "$source_dir" status --porcelain --untracked-files=normal)
  [[ -z $status ]] || {
    print -u2 "Source tree is not clean: $source_dir"
    print -u2 -- "$status"
    exit 1
  }
  if [[ -n ${OPENSN_TUO_REVISION:-} &&
        $OPENSN_TUO_REVISION != $revision ]]; then
    print -u2 "Source revision $revision does not match OPENSN_TUO_REVISION"
    exit 1
  fi
  framework_revision=$(git -C "$framework_source" rev-parse --verify HEAD^{commit})
  [[ ${#framework_revision} == 40 &&
      $framework_revision != *[^0-9a-f]* ]] || {
    print -u2 "Unable to resolve the Tuo framework revision: $framework_source"
    exit 1
  }
  local framework_status
  framework_status=$(git -C "$framework_source" status --porcelain --untracked-files=normal)
  [[ -z $framework_status ]] || {
    print -u2 "Tuo framework tree is not clean: $framework_source"
    print -u2 -- "$framework_status"
    exit 1
  }
}

configure_caliper_backend()
{
  caliper_gpu_backend=${OPENSN_TUO_CALIPER_GPU_BACKEND:-ROCM}
  case $caliper_gpu_backend in
    ROCM|NONE) ;;
    *)
      print -u2 'Tuo Caliper backend must be ROCM or NONE; CUDA/CUPTI is invalid.'
      exit 2
      ;;
  esac
  dependency_interface=legacy
  if grep -q 'OPENSN_CALIPER_GPU_BACKEND' \
    "$source_dir/tools/dependencies/CMakeLists.txt"; then
    dependency_interface=current
  fi
}

initialize_layout()
{
  require_clean_source
  load_toolchain
  configure_caliper_backend
  local revision=$(source_revision)
  [[ -n $stack_id ]] || stack_id=${revision[1,12]}
  local packages="${(j: :)python_packages} $mpi4py_package"
  local digest
  digest=$( {
    print -- 'tuo-stack-schema=2'
    print -- "stack_id=$stack_id"
    print -- 'modules=python/3.13.2 cmake/3.29.2 rocm/7.2.1 rocmcc/7.2.1-magic cray-mpich/9.1.0'
    print -- "python_packages=$packages"
    print -- "caliper_with_mpi=ON"
    print -- "caliper_gpu_backend=$caliper_gpu_backend"
    print -- "dependency_interface=$dependency_interface"
    print -- "framework_revision=$framework_revision"
    print -- "caliper_with_cupti=OFF"
    sha256sum "$script"
    find "$source_dir/tools/dependencies" -type f -print0 |
      sort -z | xargs -0 sha256sum
  } | sha256sum | awk '{print $1}' )
  stack_fingerprint=${digest[1,20]}
  stack_root=$work_root/stacks/$stack_fingerprint
  venv=$stack_root/venv
  deps_build=$stack_root/build-deps-ninja
  deps_prefix=$stack_root/deps
  deps_driver=$stack_root/dependency-driver
  deps_manifest=$stack_root/tuo-dependencies-manifest.json
  caliper_features=$stack_root/caliper-features.json
  env_file=$stack_root/env.zsh
  fingerprint_file=$stack_root/fingerprint.txt
  opensn_build=${OPENSN_TUO_BUILD:-$work_root/builds/${revision[1,12]}-$stack_fingerprint}
  build_manifest=$opensn_build/tuo-build-manifest.json
}

build_jobs()
{
  local jobs=${OPENSN_TUO_BUILD_JOBS:-}
  if [[ $jobs != <1-> ]]; then
    jobs=$(getconf _NPROCESSORS_ONLN)
    (( jobs > 84 )) && jobs=84
  fi
  print -- "$jobs"
}

clear_make_jobserver()
{
  unset MAKEFLAGS MFLAGS MAKELEVEL CMAKE_BUILD_PARALLEL_LEVEL 2>/dev/null || true
}

require_fresh_layout()
{
  [[ ! -e $stack_root ]] || {
    print -u2 "Fingerprint stack already exists: $stack_root"
    print -u2 'Choose a new OPENSN_TUO_STACK_ID for a fresh rebuild.'
    exit 1
  }
  [[ ! -e $opensn_build ]] || {
    print -u2 "OpenSn build path already exists: $opensn_build"
    print -u2 'Choose a new OPENSN_TUO_BUILD path for a fresh rebuild.'
    exit 1
  }
}

setup_fresh_venv()
{
  mkdir -p -- "$stack_root"
  python3 -m venv "$venv"
  source "$venv/bin/activate"
  python -m pip install --no-cache-dir --upgrade "${python_packages[@]}"
  MPICC=$(command -v mpicc) \
    python -m pip install --no-cache-dir --no-binary=mpi4py "$mpi4py_package"
  python -m pip check
  python - "${python_packages[@]}" "$mpi4py_package" <<'PY'
import importlib.metadata
import re
import sys

def normalized(name):
    return re.sub(r"[-_.]+", "-", name).lower()

expected = {}
for specification in sys.argv[1:]:
    name, version = specification.split("==", 1)
    expected[normalized(name)] = version
actual = {
    normalized(distribution.metadata["Name"]): distribution.version
    for distribution in importlib.metadata.distributions()
}
if actual != expected:
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    wrong = sorted(
        name
        for name in set(actual) & set(expected)
        if actual[name] != expected[name]
    )
    raise SystemExit(
        f"fresh venv differs from the lock: missing={missing}, "
        f"extra={extra}, wrong={wrong}"
    )
PY
  [[ $(gmsh --version) == $gmsh_version ]] || {
    print -u2 "Expected Gmsh $gmsh_version in the fresh environment."
    exit 1
  }
  {
    print -- "stack_fingerprint=$stack_fingerprint"
    print -- "stack_id=$stack_id"
    print -- "source_revision=$(source_revision)"
    print -- "caliper_with_mpi=ON"
    print -- "caliper_gpu_backend=$caliper_gpu_backend"
    print -- "dependency_interface=$dependency_interface"
    print -- "framework_revision=$framework_revision"
    print -- "bootstrap_sha256=$(sha256sum "$script" | awk '{print $1}')"
    print -- "caliper_with_cupti=OFF"
    print -- 'python_packages_begin'
    python -m pip freeze --all
    print -- 'python_packages_end'
  } >| "$fingerprint_file"
}

prepare_dependency_driver()
{
  local driver_source=$deps_driver/tools/dependencies
  cmake -E copy_directory "$source_dir/tools/dependencies" "$driver_source"
  cmake -E copy_directory "$source_dir/cmake" "$deps_driver/cmake"
  local input=$source_dir/tools/dependencies/CMakeLists.txt
  local output=$driver_source/CMakeLists.txt
  if grep -q -- \
    'URL_HASH SHA256=28c6e8fd940bdee9e80d1e8ae1ce0f76d6a690cbb6242d4eec115d6c0204e331' \
    "$input"; then
    sed -e 's/--download-cmake=yes //' "$input" >| "$output"
  else
    sed \
      -e 's/--download-cmake=yes //' \
      -e '/URL https:\/\/github.com\/LLNL\/Caliper\/archive\/refs\/tags\/v2.13.0.tar.gz/a\      URL_HASH SHA256=28c6e8fd940bdee9e80d1e8ae1ce0f76d6a690cbb6242d4eec115d6c0204e331' \
      "$input" >| "$output"
  fi
  if [[ $dependency_interface == legacy ]]; then
    local rocprofiler=OFF
    [[ $caliper_gpu_backend != ROCM ]] || rocprofiler=ON
    sed -i \
      "s/-DWITH_MPI=ON -DWITH_KOKKOS=OFF -DWITH_GOTCHA=OFF/-DWITH_MPI=ON -DWITH_NVTX=OFF -DWITH_CUPTI=OFF -DWITH_ROCPROFILER=$rocprofiler -DWITH_KOKKOS=OFF -DWITH_GOTCHA=OFF/" \
      "$output"
    grep -q -- "-DWITH_ROCPROFILER=$rocprofiler" "$output" || {
      print -u2 'Failed to configure Caliper features in the legacy driver.'
      exit 1
    }
  fi
  ! grep -q -- '--download-cmake=yes' "$output" || {
    print -u2 'Failed to disable the unnecessary PETSc CMake download.'
    exit 1
  }
  grep -q -- \
    'URL_HASH SHA256=28c6e8fd940bdee9e80d1e8ae1ce0f76d6a690cbb6242d4eec115d6c0204e331' \
    "$output" || {
    print -u2 'Failed to inject the verified Caliper source hash.'
    exit 1
  }
}

configure_deps()
{
  initialize_layout
  require_fresh_layout
  setup_fresh_venv
  clear_make_jobserver
  local jobs=$(build_jobs)
  mkdir -p -- "$deps_build" "$deps_prefix"
  prepare_dependency_driver
  export CMAKE_PREFIX_PATH=$deps_prefix
  export CFLAGS='-O3 -DNDEBUG'
  export CXXFLAGS='-O3 -DNDEBUG'
  local -a dependency_args=()
  if [[ $dependency_interface == current ]]; then
    dependency_args=(
      -DOPENSN_FORCE_DEPENDENCY_REBUILD=ON
      -DOPENSN_CALIPER_GPU_BACKEND="$caliper_gpu_backend"
    )
  fi
  cmake -G Ninja -S "$deps_driver/tools/dependencies" -B "$deps_build" \
    -DCMAKE_INSTALL_PREFIX="$deps_prefix" \
    -DCMAKE_BUILD_TYPE=Release \
    -DMPI_C_COMPILER="$(command -v mpicc)" \
    -DMPI_CXX_COMPILER="$(command -v mpicxx)" \
    -DPython3_EXECUTABLE="$venv/bin/python" \
    -DENABLE_CALIPER=ON \
    "${dependency_args[@]}" \
    -DJOBS="$jobs" \
    -DCMAKE_DISABLE_FIND_PACKAGE_mpicpp-lite=TRUE \
    -DCMAKE_DISABLE_FIND_PACKAGE_Boost=TRUE \
    -DCMAKE_DISABLE_FIND_PACKAGE_PETSc=TRUE \
    -DCMAKE_DISABLE_FIND_PACKAGE_HDF5=TRUE \
    -DCMAKE_DISABLE_FIND_PACKAGE_VTK=TRUE \
    -DCMAKE_DISABLE_FIND_PACKAGE_caliper=TRUE
}

write_environment()
{
  cat >| "$env_file" <<EOF
if (( ! \$+functions[module] && ! \$+commands[module] )); then
  for candidate in \${MODULESHOME:-}/init/zsh /usr/share/lmod/lmod/init/zsh /etc/profile.d/z00_lmod.sh; do
    if [[ -n \$candidate && -r \$candidate ]]; then
      source "\$candidate"
      break
    fi
  done
fi
module purge
module load python/3.13.2 cmake/3.29.2 rocm/7.2.1 rocmcc/7.2.1-magic cray-mpich/9.1.0
export CC=amdclang
export CXX=amdclang++
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_SMP_SINGLE_COPY_MODE=XPMEM
source ${venv:q}/bin/activate
source ${deps_prefix:q}/bin/set_opensn_env.sh
export PATH=${deps_prefix:q}/bin:\$PATH
export LD_LIBRARY_PATH=${deps_prefix:q}/lib:${deps_prefix:q}/lib64\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
EOF
  chmod 600 "$env_file"
}

write_caliper_features()
{
  python - "$deps_build" "$deps_prefix" "$caliper_features" \
    "$caliper_gpu_backend" <<'PY'
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

build = Path(sys.argv[1])
prefix = Path(sys.argv[2])
output = Path(sys.argv[3])
requested_backend = sys.argv[4]
caches = [
    path
    for path in build.rglob("CMakeCache.txt")
    if "caliper" in str(path.parent).lower()
]
if len(caches) != 1:
    raise SystemExit(f"expected one Caliper CMakeCache.txt, found {len(caches)}")
cache = caches[0]
entries = {}
for line in cache.read_text(errors="replace").splitlines():
    match = re.match(r"(WITH_[A-Z0-9_]+):([^=]+)=(.*)", line)
    if match:
        entries[match.group(1)] = {"type": match.group(2), "value": match.group(3)}
expected_features = {
    "WITH_MPI": "ON",
    "WITH_NVTX": "OFF",
    "WITH_CUPTI": "OFF",
    "WITH_ROCPROFILER": "ON" if requested_backend == "ROCM" else "OFF",
}
for key, expected in expected_features.items():
    entry = entries.get(key, {})
    if entry.get("type") != "BOOL" or entry.get("value") != expected:
        raise SystemExit(f"Caliper {key} was not configured as {expected}: {entry}")
query = prefix / "bin/cali-query"
result = subprocess.run(
    [str(query), "--help", "configs"],
    check=False,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
if result.returncode != 0:
    raise SystemExit(f"cannot query installed Caliper recipes: {result.stdout.strip()}")
available_recipes = sorted(
    match.group(1)
    for line in result.stdout.splitlines()
    if (match := re.match(r"^\s*([a-z0-9-]+)\s+", line))
)
required_recipes = {"runtime-report", "mpi-report"}
if requested_backend == "ROCM":
    required_recipes.add("rocm-activity-report")
missing_recipes = required_recipes - set(available_recipes)
if missing_recipes:
    raise SystemExit(
        "installed Caliper is missing recipes: " + ", ".join(sorted(missing_recipes))
    )
macro_files = []
macros = []
for path in (prefix / "include").rglob("*"):
    if not path.is_file() or "caliper" not in str(path).lower():
        continue
    text = path.read_text(errors="ignore")
    selected = [
        line.strip()
        for line in text.splitlines()
        if re.search(r"(CALIPER_HAVE|HAVE_MPI|ROCPROFILER|NVTX|CUPTI)", line)
    ]
    if selected:
        macro_files.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        macros.extend(selected)
payload = {
    "schema_version": 1,
    "cache": str(cache),
    "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
    "requested": {
        "OPENSN_CALIPER_GPU_BACKEND": requested_backend,
        "WITH_MPI": "ON",
        "WITH_NVTX": "OFF",
        "WITH_CUPTI": "OFF",
        "WITH_ROCPROFILER": expected_features["WITH_ROCPROFILER"],
    },
    "cache_features": entries,
    "installed_macro_files": macro_files,
    "installed_macros": sorted(set(macros)),
    "available_config_recipes": available_recipes,
    "config_query_output": result.stdout.splitlines(),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

write_dependencies_manifest()
{
  local modules=$(module -t list 2>&1 || true)
  python - "$deps_manifest" "$source_dir" "$(source_revision)" \
    "$stack_fingerprint" "$stack_id" "$deps_prefix" "$fingerprint_file" \
    "$caliper_features" "$env_file" "$modules" \
    "$deps_driver/tools/dependencies/CMakeLists.txt" \
    "$deps_build/CMakeCache.txt" "$script" "$framework_revision" \
    "$dependency_interface" "$caliper_gpu_backend" <<'PY'
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    output,
    source,
    revision,
    fingerprint,
    stack_id,
    prefix,
    fingerprint_file,
    caliper_features,
    environment,
    modules,
    dependency_driver,
    cmake_cache,
    bootstrap,
    framework_revision,
    dependency_interface,
    caliper_gpu_backend,
) = sys.argv[1:]

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

source_path = Path(source)
dependency_build = Path(cmake_cache).parent
inputs = sorted(
    path
    for path in (source_path / "tools/dependencies").rglob("*")
    if path.is_file()
)
payload = {
    "schema_version": 2,
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "source": str(source_path.resolve()),
    "source_revision": revision,
    "framework_revision": framework_revision,
    "dependency_interface": dependency_interface,
    "caliper_gpu_backend": caliper_gpu_backend,
    "bootstrap": str(Path(bootstrap).resolve()),
    "bootstrap_sha256": sha256(bootstrap),
    "stack_fingerprint": fingerprint,
    "stack_id": stack_id,
    "prefix": str(Path(prefix).resolve()),
    "modules": modules.splitlines(),
    "python_packages": subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines(),
    "inputs": {
        str(path.relative_to(source_path)): sha256(path)
        for path in inputs
    },
    "downloaded_archives": {
        str(path.relative_to(dependency_build)): sha256(path)
        for path in sorted(dependency_build.rglob("*"))
        if path.is_file()
        and any(
            str(path).endswith(suffix)
            for suffix in (".tar.gz", ".tar.bz2", ".tgz", ".zip")
        )
    },
    "fingerprint_file_sha256": sha256(fingerprint_file),
    "caliper_features": json.loads(Path(caliper_features).read_text()),
    "caliper_features_sha256": sha256(caliper_features),
    "environment": str(Path(environment).resolve()),
    "environment_sha256": sha256(environment),
    "dependency_driver": str(Path(dependency_driver).resolve()),
    "dependency_driver_sha256": sha256(dependency_driver),
    "cmake_cache": str(Path(cmake_cache).resolve()),
    "cmake_cache_sha256": sha256(cmake_cache),
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

build_deps()
{
  initialize_layout
  [[ -x $venv/bin/python && -r $fingerprint_file && -d $deps_build ]] || {
    print -u2 "Run '$0 configure-deps' first."
    exit 1
  }
  source "$venv/bin/activate"
  export CMAKE_PREFIX_PATH=$deps_prefix
  clear_make_jobserver
  print -- 'Building dependency projects sequentially.'
  print -- "Each project may use $(build_jobs) workers."
  cmake --build "$deps_build" --parallel 1
  [[ -r $deps_prefix/bin/set_opensn_env.sh ]] || {
    print -u2 "Dependency environment was not generated in $deps_prefix"
    exit 1
  }
  ensure_boost_config
  source "$deps_prefix/bin/set_opensn_env.sh"
  export PATH=$deps_prefix/bin:$PATH
  export LD_LIBRARY_PATH=$deps_prefix/lib:$deps_prefix/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  write_environment
  write_caliper_features
  write_dependencies_manifest
}

ensure_boost_config()
{
  python - "$deps_prefix" <<'PY'
import re
import sys
from pathlib import Path

prefix = Path(sys.argv[1]).resolve()
version_header = prefix / "include/boost/version.hpp"
if not version_header.is_file():
    raise SystemExit(f"fresh Boost version header is missing: {version_header}")
header = version_header.read_text(errors="replace")
match = re.search(r'^#define BOOST_LIB_VERSION "([0-9_]+)"$', header, re.MULTILINE)
if match is None or match.group(1) != "1_86":
    raise SystemExit(f"expected fresh Boost 1.86 headers, found {match}")

directory = prefix / "lib/cmake/Boost-1.86.0"
directory.mkdir(parents=True, exist_ok=True)
config = directory / "BoostConfig.cmake"
version = directory / "BoostConfigVersion.cmake"
if not config.is_file():
    config.write_text('''# Header-only Boost package configuration for the OpenSn dependency bundle.
get_filename_component(_BOOST_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

set(Boost_FOUND TRUE)
set(Boost_VERSION 1.86.0)
set(Boost_VERSION_STRING 1.86.0)
set(Boost_INCLUDE_DIR "${_BOOST_PREFIX}/include")
set(Boost_INCLUDE_DIRS "${Boost_INCLUDE_DIR}")

if(NOT TARGET Boost::headers)
  add_library(Boost::headers INTERFACE IMPORTED)
  set_target_properties(Boost::headers PROPERTIES
                        INTERFACE_INCLUDE_DIRECTORIES "${Boost_INCLUDE_DIR}")
endif()
if(NOT TARGET Boost::boost)
  add_library(Boost::boost INTERFACE IMPORTED)
  set_target_properties(Boost::boost PROPERTIES
                        INTERFACE_LINK_LIBRARIES Boost::headers)
endif()

set(Boost_LIBRARIES Boost::headers)
unset(_BOOST_PREFIX)
''')
if not version.is_file():
    version.write_text('''set(PACKAGE_VERSION 1.86.0)
if(PACKAGE_VERSION VERSION_LESS PACKAGE_FIND_VERSION)
  set(PACKAGE_VERSION_COMPATIBLE FALSE)
else()
  set(PACKAGE_VERSION_COMPATIBLE TRUE)
  if(PACKAGE_FIND_VERSION STREQUAL PACKAGE_VERSION)
    set(PACKAGE_VERSION_EXACT TRUE)
  endif()
endif()
''')
for path in (config, version):
    text = path.read_text(errors="replace")
    if "1.86.0" not in text:
        raise SystemExit(f"invalid fresh Boost package configuration: {path}")
PY
}

write_build_manifest()
{
  local revision=$(source_revision)
  local binary=$opensn_build/python/opensn
  local modules=$(module -t list 2>&1 || true)
  local temporary=$build_manifest.tmp-$$
  python - "$temporary" "$source_dir" "$revision" "$binary" \
    "$env_file" "$deps_manifest" "$caliper_features" "$modules" \
    "$opensn_build/CMakeCache.txt" "$stack_fingerprint" <<'PY'
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    output,
    source,
    revision,
    binary,
    environment,
    deps_manifest,
    caliper_features,
    modules,
    cmake_cache,
    stack_fingerprint,
) = sys.argv[1:]

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def output_of(command):
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout.strip() or "unavailable"

def dynamic_library_closure(executable):
    result = subprocess.run(
        ["ldd", executable],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"ldd failed for {executable}: {result.stdout.strip()}")
    libraries = {}
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if "=> not found" in stripped:
            raise SystemExit(f"unresolved dynamic library: {stripped}")
        candidate = stripped.split("=>", 1)[-1].split(" (", 1)[0].strip()
        if not candidate.startswith("/"):
            continue
        path = Path(candidate).resolve()
        if not path.is_file():
            raise SystemExit(f"dynamic library does not exist: {path}")
        libraries[str(path)] = sha256(path)
    if not any(re.match(r"libopensn\.so(?:\.|$)", Path(path).name) for path in libraries):
        raise SystemExit("OpenSn dynamic closure does not contain libopensn.so")
    return [
        {"path": path, "sha256": digest}
        for path, digest in sorted(libraries.items())
    ]

cache_entries = {}
for line in Path(cmake_cache).read_text(errors="replace").splitlines():
    if line.startswith("Boost_DIR:") and "=" in line:
        cache_entries["Boost_DIR"] = line.split("=", 1)[1]
boost_config_dir = Path(cache_entries.get("Boost_DIR", ""))
boost_config_file = boost_config_dir / "BoostConfig.cmake"
boost_version_file = boost_config_dir / "BoostConfigVersion.cmake"
if boost_config_dir.name != "Boost-1.86.0":
    raise SystemExit(f"OpenSn did not select the fresh Boost 1.86 config: {boost_config_dir}")
for path in (boost_config_file, boost_version_file):
    if not path.is_file():
        raise SystemExit(f"missing Boost package provenance file: {path}")

payload = {
    "schema_version": 2,
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "source": str(Path(source).resolve()),
    "revision": revision,
    "source_clean": True,
    "binary": str(Path(binary).resolve()),
    "binary_sha256": sha256(binary),
    "environment": str(Path(environment).resolve()),
    "environment_sha256": sha256(environment),
    "dependencies_manifest": str(Path(deps_manifest).resolve()),
    "dependencies_manifest_sha256": sha256(deps_manifest),
    "stack_fingerprint": stack_fingerprint,
    "caliper_features_manifest": str(Path(caliper_features).resolve()),
    "caliper_features": json.loads(Path(caliper_features).read_text()),
    "caliper_features_sha256": sha256(caliper_features),
    "cmake_cache": str(Path(cmake_cache).resolve()),
    "cmake_cache_sha256": sha256(cmake_cache),
    "modules": modules.splitlines(),
    "compiler": output_of(["amdclang++", "--version"]).splitlines()[0],
    "cmake": output_of(["cmake", "--version"]).splitlines()[0],
    "gmsh": output_of(["gmsh", "--version"]).splitlines()[-1],
    "mpi": output_of(["mpicxx", "--version"]).splitlines()[0],
    "caliper": output_of(["cali-query", "--version"]).splitlines()[0],
    "linked_libraries": output_of(["ldd", binary]).splitlines(),
    "linked_library_closure": dynamic_library_closure(binary),
    "boost": {
        "version": "1.86.0",
        "config": str(boost_config_file.resolve()),
        "config_sha256": sha256(boost_config_file),
        "version_config": str(boost_version_file.resolve()),
        "version_config_sha256": sha256(boost_version_file),
    },
    "hip_architecture": "gfx942",
    "build_type": "Native",
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
  mv -- "$temporary" "$build_manifest"
}

validate_opensn_dependencies()
{
  python - "$opensn_build/CMakeCache.txt" "$deps_prefix" <<'PY'
import sys
from pathlib import Path

cache = Path(sys.argv[1])
prefix = Path(sys.argv[2]).resolve()
entries = {}
for line in cache.read_text(errors="replace").splitlines():
    if line.startswith(("//", "#")) or ":" not in line or "=" not in line:
        continue
    key, remainder = line.split(":", 1)
    _, value = remainder.split("=", 1)
    entries[key] = value
required = (
    "Boost_DIR",
    "Boost_INCLUDE_DIR",
    "HDF5_DIR",
    "VTK_DIR",
    "PETSC_INCLUDE_DIR",
    "PETSC_LIBRARY",
    "caliper_DIR",
    "mpicpp-lite_DIR",
)
errors = []
for key in required:
    value = entries.get(key)
    if not value:
        errors.append(f"{key}=missing")
        continue
    try:
        Path(value).resolve().relative_to(prefix)
    except ValueError:
        errors.append(f"{key}={value}")
if errors:
    raise SystemExit(
        "OpenSn selected dependencies outside the fresh prefix: " + ", ".join(errors)
    )
boost_dir = Path(entries["Boost_DIR"]).resolve()
if boost_dir.name != "Boost-1.86.0":
    raise SystemExit(f"OpenSn did not select Boost 1.86.0: {boost_dir}")
config = boost_dir / "BoostConfig.cmake"
version_config = boost_dir / "BoostConfigVersion.cmake"
if not config.is_file() or not version_config.is_file():
    raise SystemExit(f"fresh Boost package configuration is incomplete: {boost_dir}")
if "set(Boost_VERSION 1.86.0)" not in config.read_text(errors="replace"):
    raise SystemExit(f"fresh Boost configuration has the wrong version: {config}")
PY
}

build_opensn()
{
  initialize_layout
  [[ -r $env_file && -r $deps_manifest && -r $caliper_features ]] || {
    print -u2 "Completed dependency provenance does not exist in $stack_root"
    exit 1
  }
  source "$env_file"
  clear_make_jobserver
  local pybind_dir=$(python -m pybind11 --cmakedir)
  local boost_config_dir=$deps_prefix/lib/cmake/Boost-1.86.0
  [[ -r $boost_config_dir/BoostConfig.cmake &&
     -r $boost_config_dir/BoostConfigVersion.cmake ]] || {
    print -u2 "Fresh Boost 1.86 package configuration is missing: $boost_config_dir"
    exit 1
  }
  local jobs=$(build_jobs)
  cmake -G Ninja -S "$source_dir" -B "$opensn_build" \
    -DCMAKE_BUILD_TYPE=Native \
    -DCMAKE_PREFIX_PATH="$deps_prefix;$pybind_dir" \
    -DHDF5_DIR="$deps_prefix/cmake" \
    -DVTK_DIR="$deps_prefix/lib/cmake/vtk-9.3" \
    -Dmpicpp-lite_DIR="$deps_prefix/lib/cmake/mpicpp-lite" \
    -DPETSC_DIR="$deps_prefix" \
    -DBoost_DIR="$boost_config_dir" \
    -DBOOST_ROOT="$deps_prefix" \
    -DBoost_ROOT="$deps_prefix" \
    -DBoost_NO_SYSTEM_PATHS=ON \
    -DBoost_NO_BOOST_CMAKE=OFF \
    -DBoost_INCLUDE_DIR="$deps_prefix/include" \
    -Dcaliper_DIR="$deps_prefix/share/cmake/caliper" \
    -Dpybind11_DIR="$pybind_dir" \
    -DCMAKE_HIP_ARCHITECTURES=gfx942 \
    -DOPENSN_WITH_HIP=ON \
    -DOPENSN_WITH_PYTHON=ON \
    -DOPENSN_WITH_PYTHON_MODULE=ON \
    -DPython3_EXECUTABLE="$venv/bin/python" \
    -DCMAKE_DISABLE_FIND_PACKAGE_GTest=TRUE
  validate_opensn_dependencies
  cmake --build "$opensn_build" --parallel "$jobs"
  [[ -x $opensn_build/python/opensn ]] || {
    print -u2 "OpenSn binary was not generated in $opensn_build"
    exit 1
  }
  write_build_manifest
  print -- "Build manifest: $build_manifest"
}

verify_build()
{
  initialize_layout
  [[ -r $build_manifest ]] || {
    print -u2 "Build manifest does not exist: $build_manifest"
    exit 1
  }
  python - "$build_manifest" "$(source_revision)" \
    "$opensn_build/python/opensn" "$env_file" "$deps_manifest" \
    "$caliper_features" "$opensn_build/CMakeCache.txt" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

manifest_path, revision, binary, environment, deps, caliper, cache = sys.argv[1:]
manifest = json.loads(Path(manifest_path).read_text())
dependencies = json.loads(Path(deps).read_text())

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def resolved_dynamic_paths(executable):
    result = subprocess.run(
        ["ldd", executable],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0 or "=> not found" in result.stdout:
        raise SystemExit(f"cannot resolve current dynamic closure: {result.stdout}")
    paths = set()
    for line in result.stdout.splitlines():
        candidate = line.strip().split("=>", 1)[-1].split(" (", 1)[0].strip()
        if candidate.startswith("/"):
            paths.add(str(Path(candidate).resolve()))
    return sorted(paths)

expected = {
    "revision": revision,
    "binary": str(Path(binary).resolve()),
    "binary_sha256": sha256(binary),
    "environment": str(Path(environment).resolve()),
    "environment_sha256": sha256(environment),
    "dependencies_manifest_sha256": sha256(deps),
    "caliper_features_sha256": sha256(caliper),
    "cmake_cache_sha256": sha256(cache),
}
errors = [key for key, value in expected.items() if manifest.get(key) != value]
for path_key, hash_key in (
    ("bootstrap", "bootstrap_sha256"),
    ("dependency_driver", "dependency_driver_sha256"),
    ("cmake_cache", "cmake_cache_sha256"),
):
    path = Path(dependencies.get(path_key, ""))
    if not path.is_file() or sha256(path) != dependencies.get(hash_key):
        errors.append(f"dependencies.{path_key}")
boost = manifest.get("boost", {})
if boost.get("version") != "1.86.0":
    errors.append("boost.version")
for path_key, hash_key in (
    ("config", "config_sha256"),
    ("version_config", "version_config_sha256"),
):
    path = Path(boost.get(path_key, ""))
    if not path.is_file() or sha256(path) != boost.get(hash_key):
        errors.append(f"boost.{path_key}")
closure = manifest.get("linked_library_closure", [])
if not closure:
    errors.append("linked_library_closure")
for entry in closure:
    path = Path(entry.get("path", ""))
    if not path.is_file() or sha256(path) != entry.get("sha256"):
        errors.append(f"linked_library:{path}")
if resolved_dynamic_paths(binary) != sorted(entry.get("path", "") for entry in closure):
    errors.append("linked_library:resolved_closure")
if not any(Path(entry.get("path", "")).name.startswith("libopensn.so") for entry in closure):
    errors.append("linked_library:libopensn.so")
if not manifest.get("source_clean"):
    errors.append("source_clean")
if errors:
    raise SystemExit("Invalid build manifest fields: " + ", ".join(errors))
print(f"Verified build {revision} ({expected['binary_sha256']})")
PY
}

paths()
{
  initialize_layout
  print -- "source=$source_dir"
  print -- "revision=$(source_revision)"
  print -- "stack_id=$stack_id"
  print -- "stack_fingerprint=$stack_fingerprint"
  print -- "stack=$stack_root"
  print -- "venv=$venv"
  print -- "dependencies=$deps_prefix"
  print -- "environment=$env_file"
  print -- "build=$opensn_build"
  print -- "caliper_gpu_backend=$caliper_gpu_backend"
}

(( $# == 1 )) || usage
case $1 in
  configure-deps) configure_deps ;;
  build-deps) build_deps ;;
  build-opensn) build_opensn ;;
  verify-build) verify_build ;;
  paths) paths ;;
  all)
    configure_deps
    build_deps
    build_opensn
    verify_build
    ;;
  *) usage ;;
esac
