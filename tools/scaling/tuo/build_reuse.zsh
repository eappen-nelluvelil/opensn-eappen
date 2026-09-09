#!/usr/bin/env zsh

set -eu
setopt pipe_fail
(( $# == 3 )) || { print -u2 'usage: build_reuse.zsh SOURCE REUSE_ROOT NEW_ROOT'; exit 2; }
source_dir=${1:A}
reuse_root=${2:A}
build_root=${3:A}
[[ $build_root != $reuse_root && -r $reuse_root/env.zsh ]] || exit 2
flux getattr jobid >/dev/null 2>&1 || {
  print -u2 'Run inside a Flux compute allocation.'
  exit 2
}
[[ -z $(git -C "$source_dir" status --porcelain) ]] || {
  print -u2 'Source checkout must be clean.'
  exit 2
}
source "$reuse_root/env.zsh"
unset MAKEFLAGS MFLAGS MAKELEVEL CMAKE_BUILD_PARALLEL_LEVEL
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
deps=$reuse_root/deps
overlay=$build_root/headers
venv=$build_root/venv
build=$build_root/build-opensn
mkdir -p "$build_root"
if [[ ! -x $venv/bin/python ]]; then
  python -m venv "$venv"
fi
source "$venv/bin/activate"
if [[ ! -f $build_root/python-ready ]]; then
  python -m pip install pybind11 numpy scipy matplotlib jinja2 ninja gmsh==4.15.2
  MPI4PY_BUILD_CONFIGURE=1 MPI4PY_BUILD_MPICC=$(command -v mpicc) \
    python -m pip install --no-cache-dir --no-binary=mpi4py mpi4py==4.1.2
  python -m pip check
  touch "$build_root/python-ready"
fi
python -c 'import numpy, scipy, pybind11; import mpi4py; mpi4py.rc.initialize=False; from mpi4py import MPI; print(MPI.Get_library_version())'

# Reuse compiled libraries unchanged; install the currently required MPI headers privately.
cmake -S "$source_dir/tools/dependencies" -B "$build_root/headers-build" \
  -DCMAKE_INSTALL_PREFIX="$overlay" -DPython3_EXECUTABLE="$venv/bin/python" \
  -DCMAKE_DISABLE_FIND_PACKAGE_mpicpp-lite=TRUE \
  -DENABLE_BOOST=OFF -DENABLE_PETSC=OFF -DENABLE_HDF5=OFF \
  -DENABLE_VTK=OFF -DENABLE_CALIPER=OFF
cmake --build "$build_root/headers-build" --parallel 1
# VTK/PETSc add the old prefix/include, which may also contain older MPI headers.
# A separate non-system include directory must take precedence over that prefix.
mkdir -p "$overlay/priority"
if [[ ! -e $overlay/priority/mpicpp-lite ]]; then
  ln -s "$overlay/include/mpicpp-lite" "$overlay/priority/mpicpp-lite"
fi
local_vtk=($deps/lib{,64}/cmake/vtk-*(N/))
(( ${#local_vtk} > 0 )) || { print -u2 "No private VTK in $deps"; exit 2; }
cmake -G Ninja -S "$source_dir" -B "$build" \
  -DCMAKE_BUILD_TYPE=Native -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_CXX_FLAGS="-I$overlay/priority" -DCMAKE_HIP_FLAGS="-I$overlay/priority" \
  -DCMAKE_PREFIX_PATH="$overlay;$deps" \
  -Dmpicpp-lite_DIR="$overlay/lib/cmake/mpicpp-lite" \
  -DBoost_ROOT="$deps" -DBoost_NO_SYSTEM_PATHS=ON \
  -DHDF5_DIR="$deps/cmake" -DVTK_DIR="$local_vtk[1]" \
  -DPETSC_DIR="$deps" -Dcaliper_DIR="$deps/share/cmake/caliper" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DPython3_EXECUTABLE="$venv/bin/python" \
  -DOPENSN_WITH_CUDA=OFF -DOPENSN_WITH_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_DISABLE_FIND_PACKAGE_GTest=TRUE
cmake --build "$build" --parallel "${OPENSN_TUO_BUILD_JOBS:-16}"
{
  print -r -- "source ${(q)reuse_root}/env.zsh"
  print -r -- "source ${(q)venv}/bin/activate"
  print -r -- 'unset PYTHONPATH PYTHONHOME'
  print -r -- 'export PYTHONNOUSERSITE=1'
} > "$build_root/env.zsh"
git -C "$source_dir" rev-parse HEAD > "$build/source-revision.txt"
print -- "Native build ready: $build/python/opensn"
