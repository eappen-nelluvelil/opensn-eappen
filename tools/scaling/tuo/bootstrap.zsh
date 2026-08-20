#!/bin/zsh

set -euo pipefail

source_dir=${OPENSN_SOURCE:-${0:A:h:h:h:h}}
work_root=${OPENSN_TUO_ROOT:-/usr/workspace/${USER}/opensn-gpu/opensn-tuolumne}
venv=${OPENSN_TUO_VENV:-$work_root/venv}
deps_build=${OPENSN_TUO_DEPS_BUILD:-$work_root/build-deps-ninja}
deps_prefix=${OPENSN_TUO_DEPS:-$work_root/deps}
opensn_build=${OPENSN_TUO_BUILD:-$work_root/build}
deps_driver=$work_root/dependency-driver
env_file=$work_root/env.zsh

usage()
{
  print -u2 "usage: $0 {venv|configure-deps|build-deps|build-opensn|all}"
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

setup_venv()
{
  load_toolchain
  mkdir -p -- "$work_root"
  if [[ ! -x $venv/bin/python ]]; then
    python3 -m venv "$venv"
  fi
  source "$venv/bin/activate"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install pybind11 numpy jinja2 matplotlib ninja
  MPICC=$(command -v mpicc) python -m pip install --no-binary=mpi4py mpi4py
}

prepare_dependency_driver()
{
  local driver_source=$deps_driver/tools/dependencies
  mkdir -p -- "$driver_source"
  cp -- "$source_dir/tools/dependencies/env_script.cmake" "$driver_source/"
  cmake -E copy_directory "$source_dir/cmake" "$deps_driver/cmake"

  # Tuo provides a suitable CMake; PETSc need not download and rebuild it.
  sed 's/--download-cmake=yes //' \
    "$source_dir/tools/dependencies/CMakeLists.txt" \
    >| "$driver_source/CMakeLists.txt"
}

configure_deps()
{
  setup_venv
  clear_make_jobserver
  local jobs=$(build_jobs)
  mkdir -p -- "$deps_build" "$deps_prefix"
  prepare_dependency_driver

  export CMAKE_PREFIX_PATH=$deps_prefix
  export CFLAGS='-O3 -DNDEBUG'
  export CXXFLAGS='-O3 -DNDEBUG'

  cmake -G Ninja -S "$deps_driver/tools/dependencies" -B "$deps_build" \
    -DCMAKE_INSTALL_PREFIX="$deps_prefix" \
    -DCMAKE_BUILD_TYPE=Release \
    -DMPI_C_COMPILER="$(command -v mpicc)" \
    -DMPI_CXX_COMPILER="$(command -v mpicxx)" \
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
  source "$venv/bin/activate"
  export CMAKE_PREFIX_PATH=$deps_prefix
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
    -DCMAKE_HIP_ARCHITECTURES=gfx942 \
    -DOPENSN_WITH_HIP=ON \
    -DOPENSN_WITH_PYTHON_MODULE=ON \
    -DPython3_EXECUTABLE="$venv/bin/python" \
    -DCMAKE_DISABLE_FIND_PACKAGE_GTest=TRUE
  cmake --build "$opensn_build" --parallel "$jobs"
}

(( $# == 1 )) || usage
case $1 in
  venv) setup_venv ;;
  configure-deps) configure_deps ;;
  build-deps) build_deps ;;
  build-opensn) build_opensn ;;
  all)
    configure_deps
    build_deps
    build_opensn
    ;;
  *) usage ;;
esac
