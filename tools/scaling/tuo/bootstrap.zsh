#!/bin/zsh

set -euo pipefail

source_dir=${OPENSN_SOURCE:-${0:A:h:h:h:h}}
work_root=${OPENSN_TUO_ROOT:-/usr/workspace/${USER}/opensn-gpu/cbcd-v2-update-3/gfx942}
venv=${OPENSN_TUO_VENV:-$work_root/venv}
deps_build=${OPENSN_TUO_DEPS_BUILD:-$work_root/build-deps}
deps_prefix=${OPENSN_TUO_DEPS:-$work_root/deps}
opensn_build=${OPENSN_TUO_BUILD:-$work_root/build-opensn}
env_file=$work_root/env.zsh
caliper_backend=${OPENSN_TUO_CALIPER_GPU_BACKEND:-ROCM}

usage()
{
  cat >&2 <<EOF
usage: $0 COMMAND

Commands:
  fresh            build a new venv, dependencies, and OpenSn
  configure-deps   create/update the venv and configure dependencies
  build-deps       build the configured dependencies
  build-opensn     configure and build OpenSn
  all              run configure-deps, build-deps, and build-opensn
  paths            print the selected build paths

Use a new OPENSN_TUO_ROOT with 'fresh'. Interrupted builds can be resumed with
configure-deps, build-deps, and build-opensn. Caliper's MPI and ROCm services
are enabled by default; set OPENSN_TUO_CALIPER_GPU_BACKEND=NONE only when GPU
activity profiling is not needed.
EOF
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
  unset PYTHONPATH PYTHONHOME VIRTUAL_ENV CONDA_PREFIX CMAKE_PREFIX_PATH
  unset PETSC_DIR PETSC_ARCH HDF5_ROOT HDF5_DIR VTK_DIR caliper_DIR
  unset Boost_DIR Boost_ROOT MPICPP_LITE_DIR PKG_CONFIG_PATH
  unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH LD_LIBRARY_PATH
  unset CC CXX MPICC MPICXX

  initialize_modules
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
  local path
  for path in "$venv" "$deps_build" "$deps_prefix" "$opensn_build"; do
    [[ ! -e $path ]] || {
      print -u2 "Fresh build path already exists: $path"
      print -u2 'Select a new OPENSN_TUO_ROOT, or resume with the individual commands.'
      exit 1
    }
  done
}

setup_venv()
{
  load_toolchain
  mkdir -p -- "$work_root"
  if [[ ! -x $venv/bin/python ]]; then
    python3 -m venv "$venv"
  fi
  source "$venv/bin/activate"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install \
    pybind11 numpy jinja2 matplotlib ninja gmsh==4.15.2
  MPICC=$(command -v mpicc) \
    python -m pip install --no-binary=mpi4py mpi4py
  python -m pip check
}

configure_deps()
{
  case ${caliper_backend:u} in
    ROCM|NONE) ;;
    *)
      print -u2 'OPENSN_TUO_CALIPER_GPU_BACKEND must be ROCM or NONE.'
      exit 2
      ;;
  esac

  setup_venv
  clear_make_jobserver
  local jobs=$(build_jobs)
  mkdir -p -- "$deps_build" "$deps_prefix"

  export CMAKE_PREFIX_PATH="$deps_prefix${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
  export CFLAGS='-O3 -DNDEBUG'
  export CXXFLAGS='-O3 -DNDEBUG'

  cmake -G Ninja -S "$source_dir/tools/dependencies" -B "$deps_build" \
    -DCMAKE_INSTALL_PREFIX="$deps_prefix" \
    -DCMAKE_BUILD_TYPE=Release \
    -DMPI_C_COMPILER="$(command -v mpicc)" \
    -DMPI_CXX_COMPILER="$(command -v mpicxx)" \
    -DPython3_EXECUTABLE="$venv/bin/python" \
    -DJOBS="$jobs" \
    -DENABLE_CALIPER=ON \
    -DOPENSN_FORCE_DEPENDENCY_REBUILD=ON \
    -DOPENSN_CALIPER_GPU_BACKEND="${caliper_backend:u}" \
    -DCMAKE_DISABLE_FIND_PACKAGE_mpicpp-lite=TRUE \
    -DCMAKE_DISABLE_FIND_PACKAGE_Boost=TRUE \
    -DCMAKE_DISABLE_FIND_PACKAGE_PETSc=TRUE \
    -DCMAKE_DISABLE_FIND_PACKAGE_HDF5=TRUE \
    -DCMAKE_DISABLE_FIND_PACKAGE_VTK=TRUE \
    -DCMAKE_DISABLE_FIND_PACKAGE_caliper=TRUE
}

write_environment()
{
  mkdir -p -- "$work_root"
  cat >| "$env_file" <<EOF
if (( ! \$+functions[module] && ! \$+commands[module] )); then
  for candidate in \${MODULESHOME:-}/init/zsh /usr/share/lmod/lmod/init/zsh /etc/profile.d/z00_lmod.sh; do
    if [[ -n \$candidate && -r \$candidate ]]; then
      source \$candidate
      break
    fi
  done
fi
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV CONDA_PREFIX CMAKE_PREFIX_PATH
unset PETSC_DIR PETSC_ARCH HDF5_ROOT HDF5_DIR VTK_DIR caliper_DIR
unset Boost_DIR Boost_ROOT MPICPP_LITE_DIR PKG_CONFIG_PATH
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH LD_LIBRARY_PATH
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

build_deps()
{
  load_toolchain
  [[ -x $venv/bin/python && -r $deps_build/CMakeCache.txt ]] || {
    print -u2 'Dependencies are not configured. Run configure-deps first.'
    exit 1
  }
  source "$venv/bin/activate"
  export CMAKE_PREFIX_PATH="$deps_prefix${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
  clear_make_jobserver
  print -- "Building dependency projects sequentially; each may use $(build_jobs) workers."
  cmake --build "$deps_build" --parallel 1
  [[ -r $deps_prefix/bin/set_opensn_env.sh ]] || {
    print -u2 "Dependency environment was not generated: $deps_prefix/bin/set_opensn_env.sh"
    exit 1
  }
  write_environment
}

build_opensn()
{
  load_toolchain
  [[ -r $env_file ]] || {
    print -u2 "Dependency environment does not exist: $env_file"
    exit 1
  }
  source "$env_file"
  clear_make_jobserver
  local pybind_dir=$(python -m pybind11 --cmakedir)
  local jobs=$(build_jobs)

  cmake -G Ninja -S "$source_dir" -B "$opensn_build" \
    -DCMAKE_BUILD_TYPE=Native \
    -DCMAKE_PREFIX_PATH="$deps_prefix;$pybind_dir" \
    -DHDF5_DIR="$deps_prefix/cmake" \
    -DVTK_DIR="$deps_prefix/lib/cmake/vtk-9.3" \
    -Dmpicpp-lite_DIR="$deps_prefix/lib/cmake/mpicpp-lite" \
    -DPETSC_DIR="$deps_prefix" \
    -DBoost_DIR="$deps_prefix/lib/cmake/Boost-1.86.0" \
    -DBOOST_ROOT="$deps_prefix" \
    -DBoost_ROOT="$deps_prefix" \
    -DBoost_NO_SYSTEM_PATHS=ON \
    -Dcaliper_DIR="$deps_prefix/share/cmake/caliper" \
    -Dpybind11_DIR="$pybind_dir" \
    -DCMAKE_HIP_ARCHITECTURES=gfx942 \
    -DOPENSN_WITH_HIP=ON \
    -DOPENSN_WITH_PYTHON=ON \
    -DOPENSN_WITH_PYTHON_MODULE=ON \
    -DPython3_EXECUTABLE="$venv/bin/python" \
    -DCMAKE_DISABLE_FIND_PACKAGE_GTest=TRUE
  cmake --build "$opensn_build" --parallel "$jobs"
  [[ -x $opensn_build/python/opensn ]] || {
    print -u2 "OpenSn binary was not generated in $opensn_build"
    exit 1
  }
  print -- "OpenSn build is ready: $opensn_build/python/opensn"
}

paths()
{
  print -- "source=$source_dir"
  print -- "root=$work_root"
  print -- "venv=$venv"
  print -- "dependencies=$deps_prefix"
  print -- "environment=$env_file"
  print -- "build=$opensn_build"
  print -- "caliper_gpu_backend=${caliper_backend:u}"
}

(( $# == 1 )) || usage
case $1 in
  fresh)
    require_fresh_layout
    configure_deps
    build_deps
    build_opensn
    ;;
  configure-deps) configure_deps ;;
  build-deps) build_deps ;;
  build-opensn) build_opensn ;;
  all)
    configure_deps
    build_deps
    build_opensn
    ;;
  paths) paths ;;
  *) usage ;;
esac
