#!/usr/bin/env zsh

set -eu
setopt pipe_fail

PROGRAM=$0
SCRIPT_DIR=${0:A:h}
SOURCE_ROOT=${OPENSN_DANE_SOURCE:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}
TOOLS_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
DEFAULT_MODULES='StdEnv python/3.14.6 git/2.46.2 cmake/4.4.3 clang/19.1.3-magic openmpi/4.1.2'

usage()
{
  print -u2 -- "Usage: $PROGRAM {setup|setup-here|paths}"
  print -u2 -- ""
  print -u2 -- "Required environment:"
  print -u2 -- "  OPENSN_DANE_BANK         Slurm account/bank (setup only)"
  print -u2 -- ""
  print -u2 -- "Optional overrides:"
  print -u2 -- "  OPENSN_DANE_WORK_ROOT    Default: /usr/workspace/\$USER/opensn-dane-cbc-scaling"
  print -u2 -- "  OPENSN_DANE_TOOLCHAIN    Toolchain tag (default: clang19-openmpi412-python314-1)"
  print -u2 -- "  OPENSN_DANE_MODULES      Modules loaded after module purge"
  print -u2 -- "  OPENSN_DANE_BUILD_JOBS   Package/build parallelism (default: 16)"
  print -u2 -- "  OPENSN_DANE_SETUP_TIME   pdebug allocation limit (default: 01:00:00)"
}

require_command()
{
  command -v "$1" >/dev/null 2>&1 || {
    print -u2 -- "ERROR: required command not found: $1"
    return 1
  }
}

set_paths()
{
  export OPENSN_DANE_WORK_ROOT=${OPENSN_DANE_WORK_ROOT:-/usr/workspace/$USER/opensn-dane-cbc-scaling}
  export OPENSN_DANE_TOOLCHAIN=${OPENSN_DANE_TOOLCHAIN:-clang19-openmpi412-python314-1}
  export OPENSN_DANE_TOOLCHAIN_ROOT=$OPENSN_DANE_WORK_ROOT/toolchains/$OPENSN_DANE_TOOLCHAIN
  export OPENSN_DANE_DEPS_BUILD=$OPENSN_DANE_TOOLCHAIN_ROOT/dependencies-build
  export OPENSN_DANE_DEPS_PREFIX=$OPENSN_DANE_TOOLCHAIN_ROOT/dependencies
  export OPENSN_DANE_VENV=$OPENSN_DANE_TOOLCHAIN_ROOT/venv
  export OPENSN_DANE_ENVIRONMENT=${OPENSN_DANE_ENVIRONMENT:-$OPENSN_DANE_TOOLCHAIN_ROOT/opensn-dane-env.sh}
}

show_paths()
{
  set_paths
  print -- "source=$SOURCE_ROOT"
  print -- "work=$OPENSN_DANE_WORK_ROOT"
  print -- "toolchain=$OPENSN_DANE_TOOLCHAIN_ROOT"
  print -- "dependencies_build=$OPENSN_DANE_DEPS_BUILD"
  print -- "dependencies_prefix=$OPENSN_DANE_DEPS_PREFIX"
  print -- "venv=$OPENSN_DANE_VENV"
  print -- "environment=$OPENSN_DANE_ENVIRONMENT"
  print -- "modules=${OPENSN_DANE_MODULES:-$DEFAULT_MODULES}"
  print -- "build_jobs=${OPENSN_DANE_BUILD_JOBS:-16}"
}

load_modules()
{
  if [[ -r /usr/share/lmod/lmod/init/zsh ]]; then
    source /usr/share/lmod/lmod/init/zsh
  fi
  require_command module
  module --force purge
  local module_name
  for module_name in ${(z)${OPENSN_DANE_MODULES:-$DEFAULT_MODULES}}; do
    module load "$module_name"
  done
  export OMPI_CC=clang OMPI_CXX=clang++
  unset MPICH_GPU_SUPPORT_ENABLED MPICH_SMP_SINGLE_COPY_MODE
  if ! command -v flex >/dev/null 2>&1; then
    module load flex
  fi
}

check_cmake_version()
{
  local version=$(cmake --version | awk 'NR == 1 {print $3}')
  python3 - "$version" <<'PY'
import sys

version = tuple(int(item) for item in sys.argv[1].split(".")[:2])
if version < (3, 29):
    raise SystemExit(f"OpenSn requires CMake 3.29 or newer; found {sys.argv[1]}")
PY
}

write_environment()
{
  local loaded_modules=$1
  local mpi_cc=$2
  local mpi_cxx=$3
  local mpi_library_path=$4
  local module_commands=""
  local module_name
  for module_name in ${(s.:.)loaded_modules}; do
    module_commands+="module load ${(q)module_name}"$'\n'
  done

  {
    print -- '#!/usr/bin/env bash'
    print -- 'set -euo pipefail'
    print
    print -- 'if [[ -r /usr/share/lmod/lmod/init/bash ]]; then'
    print -- '  source /usr/share/lmod/lmod/init/bash'
    print -- 'fi'
    print -- 'module --force purge'
    print -n -- "$module_commands"
    print
    print -- "source ${(q)OPENSN_DANE_VENV}/bin/activate"
    print -- 'unset PYTHONPATH PETSC_ARCH'
    print -- 'export PYTHONNOUSERSITE=1'
    print -- 'export OMPI_CC=clang OMPI_CXX=clang++'
    print -- 'unset MPICH_GPU_SUPPORT_ENABLED MPICH_SMP_SINGLE_COPY_MODE'
    print -- "export CC=${(q)mpi_cc}"
    print -- "export CXX=${(q)mpi_cxx}"
    print -- "export CMAKE_PREFIX_PATH=${(q)OPENSN_DANE_DEPS_PREFIX}"
    print -- "export PETSC_DIR=${(q)OPENSN_DANE_DEPS_PREFIX}"
    print -- "export HDF5_ROOT=${(q)OPENSN_DANE_DEPS_PREFIX}"
    print -- "export VTK_ROOT=${(q)OPENSN_DANE_DEPS_PREFIX}"
    print -- "export Boost_ROOT=${(q)OPENSN_DANE_DEPS_PREFIX}"
    print -- "export caliper_ROOT=${(q)OPENSN_DANE_DEPS_PREFIX}"
    print -- "export GTest_ROOT=${(q)OPENSN_DANE_DEPS_PREFIX}"
    print -- "export PATH=${(q)OPENSN_DANE_DEPS_PREFIX}/bin:${(q)OPENSN_DANE_VENV}/bin:\$PATH"
    print -n -- "export LD_LIBRARY_PATH=${(q)OPENSN_DANE_DEPS_PREFIX}/lib:"
    print -- "${(q)OPENSN_DANE_DEPS_PREFIX}/lib64:${(q)mpi_library_path}\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
    print -n -- \
      "export PKG_CONFIG_PATH=${(q)OPENSN_DANE_DEPS_PREFIX}/lib/pkgconfig:"
    print -- "${(q)OPENSN_DANE_DEPS_PREFIX}/lib64/pkgconfig"
    print -- "export OPENSN_DANE_GMSH=${(q)OPENSN_DANE_VENV}/bin/gmsh"
  } > "$OPENSN_DANE_ENVIRONMENT"
  chmod 700 "$OPENSN_DANE_ENVIRONMENT"
}

setup_here()
{
  set_paths
  load_modules

  local command_name
  for command_name in clang clang++ mpicc mpicxx cmake python3 git make flex; do
    require_command "$command_name"
  done
  check_cmake_version

  local build_jobs=${OPENSN_DANE_BUILD_JOBS:-16}
  local mpi_cc=$(command -v mpicc)
  local mpi_cxx=$(command -v mpicxx)
  local loaded_modules=${LOADEDMODULES:-}

  # Do not allow compiler variables inherited from the login shell to select a
  # different ABI for dependencies whose ExternalProject recipes rely on CC
  # and CXX rather than explicit CMake compiler arguments.
  export CC=$mpi_cc
  export CXX=$mpi_cxx

  mkdir -p \
    "$OPENSN_DANE_TOOLCHAIN_ROOT" \
    "$OPENSN_DANE_DEPS_BUILD" \
    "$OPENSN_DANE_DEPS_PREFIX"

  print -- "Compiler: $(command -v clang++)"
  mpicxx --showme:command
  local mpi_version
  mpi_version=$(mpicxx --showme:version) || return 1
  print -- "$mpi_version"
  [[ $(clang -dumpversion) == 19.1.3 ]] || {
    print -u2 'Expected Clang 19.1.3.'
    return 1
  }
  print -r -- "$mpi_version" | grep -Eq '(^|[^0-9.])4\.1\.2([^0-9.]|$)' || {
    print -u2 'Expected OpenMPI 4.1.2 from mpicxx --showme:version.'
    return 1
  }
  print -- "MPI C wrapper: $mpi_cc"
  print -- "MPI C++ wrapper: $mpi_cxx"
  print -- "CMake: $(command -v cmake)"
  print -- "Python: $(command -v python3)"
  print -- "Modules: $loaded_modules"

  local mpi_libdirs_text
  mpi_libdirs_text=$("$mpi_cc" --showme:libdirs) || return 1
  local -a mpi_library_dirs
  mpi_library_dirs=(${=mpi_libdirs_text})
  local mpi_library_path=${(j.:.)mpi_library_dirs}
  [[ -n $mpi_library_path ]] || {
    print -u2 'The MPI compiler wrapper did not report its library directories.'
    return 1
  }
  export LD_LIBRARY_PATH="$mpi_library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  unset PYTHONPATH
  export PYTHONNOUSERSITE=1
  print -- "MPI library directories: $mpi_library_path"

  if [[ ! -x $OPENSN_DANE_VENV/bin/python ]]; then
    python3 -m venv "$OPENSN_DANE_VENV"
  fi
  source "$OPENSN_DANE_VENV/bin/activate"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install \
    pybind11 numpy scipy matplotlib jinja2 pyyaml nbconvert gmsh
  local mpi_build_log=$OPENSN_DANE_TOOLCHAIN_ROOT/mpi4py-build.log
  local -a mpi_link_command
  mpi_link_command=("$mpi_cc")
  local mpi_directory
  for mpi_directory in "${mpi_library_dirs[@]}"; do
    mpi_link_command+=("-L$mpi_directory" "-Wl,-rpath,$mpi_directory")
  done
  local mpi_linker=${(j: :)${(q)mpi_link_command}}
  print -- "mpi4py linker: $mpi_linker"
  print -- "Building mpi4py with exhaustive MPI checks; log=$mpi_build_log"
  unset MPI4PY_BUILD_MPICFG MPICFG
  if ! env MPI4PY_BUILD_MPICC="$mpi_cc" MPI4PY_BUILD_MPILD="$mpi_linker" \
      MPI4PY_BUILD_BACKEND=setuptools MPI4PY_BUILD_CONFIGURE=1 \
      python -m pip install --verbose --no-cache-dir --no-binary=mpi4py \
        --force-reinstall --no-deps mpi4py==4.1.2 > "$mpi_build_log" 2>&1; then
    tail -n 80 "$mpi_build_log"
    return 1
  fi
  if ! python - "${mpi_library_dirs[@]}" <<'PY' > "$OPENSN_DANE_TOOLCHAIN_ROOT/mpi4py-linkage.txt"
import importlib.util
from pathlib import Path
import re
import subprocess
import sys

extension = importlib.util.find_spec("mpi4py.MPI").origin
print(f"mpi4py_extension={extension}", flush=True)
linkage = subprocess.check_output(["ldd", extension], text=True)
print(linkage, flush=True)
expected = {Path(directory, "libmpi.so").resolve() for directory in sys.argv[1:]
            if Path(directory, "libmpi.so").is_file()}
resolved = re.findall(r"\blibmpi\.so(?:\.\d+)*\s+=>\s+(\S+)", linkage)
if len(resolved) != 1 or Path(resolved[0]).resolve() not in expected:
    raise SystemExit("mpi4py did not link to the selected compiler wrapper's MPI library.")
print("MPI linkage matches the selected compiler wrapper.")
PY
  then
    sed -n '1,60p' "$OPENSN_DANE_TOOLCHAIN_ROOT/mpi4py-linkage.txt"
    return 1
  fi
  sed -n '1,60p' "$OPENSN_DANE_TOOLCHAIN_ROOT/mpi4py-linkage.txt"
  python - <<'PY'
import gmsh
import jinja2
import matplotlib
import mpi4py
import numpy
import scipy
from mpi4py import MPI

print("Python imports succeeded.")
print(MPI.Get_library_version())
PY
  require_command gmsh
  gmsh --version

  unset PETSC_ARCH PETSC_DIR HDF5_ROOT VTK_DIR VTK_ROOT \
    Boost_DIR Boost_ROOT caliper_DIR caliper_ROOT GTest_DIR GTest_ROOT
  export CMAKE_PREFIX_PATH=$OPENSN_DANE_DEPS_PREFIX

  cmake \
    -S "$TOOLS_ROOT/tools/dependencies" \
    -B "$OPENSN_DANE_DEPS_BUILD" \
    -DCMAKE_INSTALL_PREFIX="$OPENSN_DANE_DEPS_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$mpi_cc" \
    -DCMAKE_CXX_COMPILER="$mpi_cxx" \
    -DPython3_EXECUTABLE="$OPENSN_DANE_VENV/bin/python" \
    -DJOBS="$build_jobs" \
    -DENABLE_BOOST=ON \
    -DENABLE_PETSC=ON \
    -DENABLE_HDF5=ON \
    -DENABLE_VTK=ON \
    -DENABLE_CALIPER=ON \
    -DCMAKE_DISABLE_FIND_PACKAGE_mpicpp-lite=TRUE \
    -DCMAKE_DISABLE_FIND_PACKAGE_Boost=TRUE \
    -DCMAKE_DISABLE_FIND_PACKAGE_PETSc=TRUE \
    -DCMAKE_DISABLE_FIND_PACKAGE_HDF5=TRUE \
    -DCMAKE_DISABLE_FIND_PACKAGE_VTK=TRUE \
    -DCMAKE_DISABLE_FIND_PACKAGE_caliper=TRUE \
    -DCMAKE_FIND_USE_PACKAGE_REGISTRY=FALSE \
    -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=FALSE

  # External projects use bounded internal parallelism; keeping the outer
  # build serial prevents nested make processes from exhausting descriptors.
  cmake --build "$OPENSN_DANE_DEPS_BUILD" --parallel 1

  local gtest_source=$OPENSN_DANE_TOOLCHAIN_ROOT/sources/googletest-1.15.2
  local gtest_build=$OPENSN_DANE_TOOLCHAIN_ROOT/googletest-build
  mkdir -p "$OPENSN_DANE_TOOLCHAIN_ROOT/sources"
  if [[ ! -d $gtest_source/.git ]]; then
    if [[ -e $gtest_source ]]; then
      print -u2 -- "ERROR: incomplete GoogleTest source exists at $gtest_source"
      return 1
    fi
    git clone --depth 1 --branch v1.15.2 \
      https://github.com/google/googletest.git "$gtest_source"
  fi
  cmake \
    -S "$gtest_source" \
    -B "$gtest_build" \
    -DCMAKE_INSTALL_PREFIX="$OPENSN_DANE_DEPS_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="$(command -v clang++)" \
    -DBUILD_GMOCK=ON \
    -DINSTALL_GTEST=ON
  cmake --build "$gtest_build" --parallel "$build_jobs"
  cmake --install "$gtest_build"

  write_environment "$loaded_modules" "$mpi_cc" "$mpi_cxx" "$mpi_library_path"
  source "$OPENSN_DANE_ENVIRONMENT"

  local sha=$(git -C "$SOURCE_ROOT" rev-parse HEAD)
  local preflight=$OPENSN_DANE_TOOLCHAIN_ROOT/build-opensn-${sha[1,9]}-native
  cmake \
    -S "$SOURCE_ROOT" \
    -B "$preflight" \
    -DCMAKE_BUILD_TYPE=Native \
    -DOPENSN_WITH_CUDA=OFF \
    -DOPENSN_WITH_HIP=OFF \
    -DOPENSN_WITH_SYCL=OFF
  cmake --build "$preflight" --parallel "$build_jobs"
  grep -qx 'CMAKE_BUILD_TYPE:STRING=Native' "$preflight/CMakeCache.txt"
  [[ -x $preflight/python/opensn ]]

  print
  print -- "Dane toolchain is ready."
  print -- "environment=$OPENSN_DANE_ENVIRONMENT"
  print -- "binary=$preflight/python/opensn"
}

setup()
{
  set_paths
  : ${OPENSN_DANE_BANK:?Set OPENSN_DANE_BANK to the Slurm account/bank}
  if [[ -n ${SLURM_JOB_ID:-} ]]; then
    setup_here
    return
  fi
  require_command salloc
  require_command srun
  salloc \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${OPENSN_DANE_BUILD_JOBS:-16}" \
    --partition=pdebug \
    --account="$OPENSN_DANE_BANK" \
    --exclusive \
    --time="${OPENSN_DANE_SETUP_TIME:-01:00:00}" \
    srun \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task="${OPENSN_DANE_BUILD_JOBS:-16}" \
      zsh "$PROGRAM" setup-here
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

case $1 in
  setup)
    setup
    ;;
  setup-here)
    setup_here
    ;;
  paths)
    show_paths
    ;;
  *)
    usage
    exit 2
    ;;
esac
