#!/bin/zsh

set -euo pipefail

source_dir=${OPENSN_SOURCE:-${0:A:h:h:h:h}}
work_root=${OPENSN_DANE_ROOT:-/usr/workspace/${USER}/opensn-dane/cbc-cycles-update}
venv=${OPENSN_DANE_VENV:-$work_root/venv}
deps_build=${OPENSN_DANE_DEPS_BUILD:-$work_root/build-deps-ninja}
deps_prefix=${OPENSN_DANE_DEPS:-$work_root/deps}
opensn_build=${OPENSN_DANE_BUILD:-$work_root/build}
env_file=$work_root/env.zsh
deps_driver=$work_root/dependency-driver

usage()
{
  print -u2 "usage: $0 {venv|configure-deps|build-deps|build-opensn|all}"
  exit 2
}

module_init_script()
{
  if [[ -n ${MODULESHOME:-} && -r $MODULESHOME/init/zsh ]]; then
    print -- "$MODULESHOME/init/zsh"
  elif [[ -n ${LMOD_CMD:-} && -r ${LMOD_CMD:h:h}/init/zsh ]]; then
    print -- "${LMOD_CMD:h:h}/init/zsh"
  elif [[ -r /usr/share/lmod/lmod/init/zsh ]]; then
    print -- /usr/share/lmod/lmod/init/zsh
  elif [[ -r /etc/profile.d/z00_lmod.sh ]]; then
    print -- /etc/profile.d/z00_lmod.sh
  else
    return 1
  fi
}

initialize_modules()
{
  (( $+functions[module] || $+commands[module] )) && return

  local init_script
  init_script=$(module_init_script) || {
    print -u2 'Unable to initialize the environment-modules command.'
    exit 1
  }
  source "$init_script"
}

load_toolchain()
{
  initialize_modules
  module purge
  module load \
    python/3.13.2 \
    git/2.46.2 \
    cmake/3.30.5 \
    clang/19.1.3-magic \
    openmpi/4.1.2

  export CC=clang
  export CXX=clang++
  export OMPI_CC=clang
  export OMPI_CXX=clang++
}

require_allocation()
{
  if [[ -z ${SLURM_JOB_ID:-} ]]; then
    print -u2 'Run dependency and OpenSn compilation inside a Dane allocation.'
    print -u2 'Example: salloc -N 1 -p pdebug --exclusive -t 01:00:00'
    exit 1
  fi
}

build_jobs()
{
  local jobs=${OPENSN_DANE_BUILD_JOBS:-${SLURM_CPUS_ON_NODE:-}}
  jobs=${jobs%%\(*}
  if [[ $jobs != <1-> ]]; then
    jobs=$(getconf _NPROCESSORS_ONLN)
  fi
  print -- "$jobs"
}

clear_make_jobserver()
{
  # ExternalProject launches nested builds that must not inherit a stale GNU Make jobserver.
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
  python -m pip install pybind11 numpy jinja2 matplotlib llnl-hatchet ninja
  MPICC=$(command -v mpicc) python -m pip install --no-binary=mpi4py mpi4py
}

prepare_dependency_driver()
{
  local driver_source=$deps_driver/tools/dependencies
  mkdir -p -- "$driver_source"
  cp -- "$source_dir/tools/dependencies/env_script.cmake" "$driver_source/"
  cmake -E copy_directory "$source_dir/cmake" "$deps_driver/cmake"

  # CMake 3.30.5 is provided by Dane; do not make PETSc build another copy.
  sed 's/--download-cmake=yes //' \
    "$source_dir/tools/dependencies/CMakeLists.txt" \
    >| "$driver_source/CMakeLists.txt"
  if grep -q -- '--download-cmake' "$driver_source/CMakeLists.txt"; then
    print -u2 'Unable to disable PETSc CMake download in the private dependency driver.'
    exit 1
  fi
}

configure_deps()
{
  require_allocation
  setup_venv
  clear_make_jobserver
  local jobs=$(build_jobs)
  mkdir -p -- "$deps_build" "$deps_prefix"
  export CMAKE_PREFIX_PATH=$deps_prefix
  export CFLAGS='-O3 -DNDEBUG'
  export CXXFLAGS='-O3 -DNDEBUG'
  prepare_dependency_driver

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
  local init_script
  init_script=$(module_init_script) || {
    print -u2 'Unable to locate the Lmod initialization script.'
    exit 1
  }
  mkdir -p -- "$work_root"
  cat >| "$env_file" <<EOF
if (( ! \$+functions[module] && ! \$+commands[module] )); then
  source ${init_script:q}
fi
module purge
module load python/3.13.2 git/2.46.2 cmake/3.30.5 clang/19.1.3-magic openmpi/4.1.2
export CC=clang
export CXX=clang++
export OMPI_CC=clang
export OMPI_CXX=clang++
source ${venv:q}/bin/activate
source ${deps_prefix:q}/bin/set_opensn_env.sh
export PATH=${deps_prefix:q}/bin:\$PATH
export LD_LIBRARY_PATH=${deps_prefix:q}/lib:${deps_prefix:q}/lib64\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
EOF
  chmod 600 "$env_file"
}

build_deps()
{
  require_allocation
  load_toolchain
  source "$venv/bin/activate"
  export CMAKE_PREFIX_PATH=$deps_prefix
  clear_make_jobserver
  print -- "Building dependency projects sequentially; each project may use up to $(build_jobs) workers."
  cmake --build "$deps_build" --parallel 1
  [[ -r $deps_prefix/bin/set_opensn_env.sh ]] || {
    print -u2 "Dependency environment was not generated: $deps_prefix"
    exit 1
  }
  write_environment
}

build_opensn()
{
  require_allocation
  [[ -r $env_file ]] || {
    print -u2 "Build the dependencies first; missing $env_file"
    exit 1
  }
  source "$env_file"
  clear_make_jobserver
  local jobs=$(build_jobs)
  cmake -U 'GTest_*' -U 'GTEST_*' -S "$source_dir" -B "$opensn_build" \
    -DCMAKE_BUILD_TYPE=Native \
    -DCMAKE_DISABLE_FIND_PACKAGE_GTest=TRUE \
    -DOPENSN_WITH_PYTHON_MODULE=ON \
    -DCMAKE_PREFIX_PATH="$deps_prefix;$(python -m pybind11 --cmakedir)"
  cmake --build "$opensn_build" --parallel "$jobs"
  [[ -x $opensn_build/python/opensn ]] || {
    print -u2 "OpenSn executable was not generated: $opensn_build/python/opensn"
    exit 1
  }
  print -- "OpenSn executable: $opensn_build/python/opensn"
}

command=${1:-}
case $command in
  venv)
    setup_venv
    ;;
  configure-deps)
    configure_deps
    ;;
  build-deps)
    build_deps
    ;;
  build-opensn)
    build_opensn
    ;;
  all)
    configure_deps
    build_deps
    build_opensn
    ;;
  *)
    usage
    ;;
esac
