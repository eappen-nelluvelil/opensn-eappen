OpenSn Installation
===================

Install Development Tools
-------------------------

Before installing OpenSn, you must have the following packages installed on
your system:

1. A recent version of ``clang++``/``g++`` that supports C++20
2. Python 3.9+ with pybind11 and ``pip``
3. Git version control system
4. CMake v3.29+
5. MPI (OpenMPI, MPICH, and MVAPICH have been tested)
6. ``flex`` (required by the PTSCOTCH component of PETSc)
7. Pandoc and Doxygen (only if you plan to build documentation)

You can install these packages using your system package manager:

.. tab-set::

   .. tab-item:: Linux (Ubuntu)

      .. code-block:: shell

         sudo apt install build-essential python3-dev python3-pip \
             git cmake libopenmpi-dev flex doxygen pandoc
         export NPROC=$(nproc)

   .. tab-item:: macOS

      .. code-block:: shell

         brew install gcc python git cmake open-mpi flex doxygen pandoc
         export NPROC=$(sysctl -n hw.ncpu)

The following additional packages will be installed by the OpenSn dependency 
build (if not found on your system):

1. PETSc 3.17.0+
2. Boost 1.86+
3. HDF5 1.14+
4. VTK 9.3.0+
5. Caliper 2.14+

Virtual environments (recommended)
----------------------------------

Using a virtual environment keeps Python dependencies isolated from your system
packages and makes it easier to cleanly uninstall or upgrade later.

.. code-block:: shell

   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip

The environment must be activated in each new shell session before running
``pip`` or OpenSn's Python tools.

.. code-block:: shell

   source .venv/bin/activate
   python -m pip install pybind11

To leave the environment, run:

.. code-block:: shell

   deactivate



Clone OpenSn
------------

.. important::

   If you want to contribute to **OpenSn**, it is strongly recommended that you
   first fork the **OpenSn** repository then clone your fork.

To clone the **OpenSn** repository:

.. code-block:: shell

   git clone https://github.com/Open-Sn/opensn.git

To clone your fork of **OpenSn**:

.. code-block:: shell

   git clone https://github.com/<username>/opensn.git

Install dependencies
--------------------

You can install all required dependencies for **OpenSn** into a dedicated
directory using **CMake**:

.. code-block:: shell

   cmake -S /path/to/opensn/tools/dependencies \
         -B build_deps \
         -DCMAKE_INSTALL_PREFIX=/path/to/dependencies/directory \
         -DOPENSN_FORCE_DEPENDENCY_BUILDS=ON
   cmake --build build_deps --parallel $NPROC
   rm -rf build_deps

During this step, **CMake** automatically searches for compatible system
installations of the required packages. Any missing dependencies will be
downloaded, built, and installed into the specified directory.
``OPENSN_FORCE_DEPENDENCY_BUILDS`` gives a reproducible prefix by ignoring
compatible packages from the current shell environment.

When the process completes, a helper script named ``set_opensn_env.sh``
is generated in the installation directory:

.. code-block:: shell

   /path/to/dependencies/directory/bin/set_opensn_env.sh

This script updates your environment so that future **CMake** builds can
locate the installed libraries and headers. To make these settings
persistent, add the following line to your shell's startup file (for
example, ``~/.bashrc`` or ``~/.zshrc``):

.. code-block:: shell

   source /path/to/dependencies/directory/bin/set_opensn_env.sh

Sourcing this script ensures that **OpenSn** uses the correct versions
of all required dependencies. This setup is recommended for both end
users and developers working on the OpenSn code base.

Caliper can also be built with GPU profiling support. Enable CUDA/CUPTI
profiling when building dependencies for NVIDIA GPUs:

.. code-block:: shell

   cmake -S /path/to/opensn/tools/dependencies \
         -B build_deps \
         -DCMAKE_INSTALL_PREFIX=/path/to/dependencies/directory \
         -DOPENSN_FORCE_DEPENDENCY_BUILDS=ON \
         -DOPENSN_CALIPER_WITH_CUDA=ON
   cmake --build build_deps --parallel $NPROC

For ROCm/HIP systems, use the rocprofiler backend when it is available:

.. code-block:: shell

   cmake -S /path/to/opensn/tools/dependencies \
         -B build_deps \
         -DCMAKE_INSTALL_PREFIX=/path/to/dependencies/directory \
         -DOPENSN_FORCE_DEPENDENCY_BUILDS=ON \
         -DOPENSN_CALIPER_ROCM_BACKEND=rocprofiler
   cmake --build build_deps --parallel $NPROC

After installing a new dependency prefix, source its generated environment
script before configuring OpenSn:

.. code-block:: shell

   source /path/to/dependencies/directory/bin/set_opensn_env.sh

If you are switching from an older dependency prefix in the same shell, first
clear stale CMake package search paths:

.. code-block:: shell

   unset CMAKE_PREFIX_PATH
   hash -r

Configure and build OpenSn
--------------------------

**OpenSn** provides a Python interface. It is available in two formats: a
console application ``opensn`` and a Python module ``pyopensn``.

Classes and functions in the Python interface are detailed in :ref:`pyapi`.

.. attention::

   The console and the module are **not compatible** with each other. Attempting
   to import the module within the console will result in an import error. Users
   should select one approach and maintain consistent coding style throughout.

Console application
^^^^^^^^^^^^^^^^^^^

To compile the console application:

.. code-block:: shell

   cmake -S . -B build -DCMAKE_BUILD_TYPE=Native
   cmake --build build --parallel $NPROC

``Native`` enables architecture-specific host optimizations for the machine
where OpenSn is built. Use ``Release`` instead if the build node is not
representative of the compute nodes where the executable will run.

CUDA and HIP builds should also specify the target device architecture:

.. code-block:: shell

   cmake -S . -B build-cuda \
         -DCMAKE_BUILD_TYPE=Native \
         -DOPENSN_WITH_CUDA=ON \
         -DCMAKE_CUDA_ARCHITECTURES=<cuda-architecture>

   cmake -S . -B build-hip \
         -DCMAKE_BUILD_TYPE=Native \
         -DOPENSN_WITH_HIP=ON \
         -DCMAKE_HIP_ARCHITECTURES=<hip-architecture>

.. danger::

   In the console application, all classes and functions are implicitly imported
   into the ``__main__`` module at startup. Therefore, omit submodule prefixes
   when referring to class or function names. Additionally, avoid redefining any
   **OpenSn** class or function names to prevent naming conflicts.

Module
^^^^^^

To compile the module and install in the Python ``site-packages`` path:

.. code-block:: bash

   pip install .

For developers, it is recommended to use the following command to install the
additional packages required for running regression tests:

.. code-block:: bash

   pip install .[dev]

.. tip::

   Unlike the console, the Python interface is fully compatible with ``mpi4py``.
   Both **OpenSn** and ``mpi4py`` share the same MPI communicator. Therefore,
   the Python module can be used in scripts that incorporate other tasks using
   ``mpi4py``.

Run regression tests
--------------------

Run the regression tests to verify your installation:

.. code-block:: shell

   cd /path/to/opensn
   test/run_tests -d test/python -j$NPROC -v 1 -w 3

.. attention::

   Regression tests require both the console and the module. This can be
   achieved with:

   .. code-block:: shell

      cmake -DOPENSN_WITH_PYTHON_MODULE=ON ..

Caliper profiling
-----------------

The console executable accepts raw Caliper configuration strings and a small
set of OpenSn presets:

.. code-block:: shell

   opensn --caliper -i input.py
   opensn --caliper-preset mpi -i input.py
   opensn --caliper-preset cuda -i input.py
   opensn --caliper-preset hip -i input.py

``--caliper`` without an argument prints an inclusive runtime report for
annotated OpenSn regions. The ``mpi`` preset adds MPI timing, message counts,
message sizes, and memory high-water marks. The ``cuda`` and ``hip`` presets
require Caliper to be built with the corresponding GPU profiling backend.

For machine-readable output, use:

.. code-block:: shell

   opensn --caliper-preset profile -i input.py
   opensn --caliper-preset cuda-profile -i input.py
   opensn --caliper-preset hip-profile -i input.py

Raw Caliper configurations remain supported:

.. code-block:: shell

   opensn --caliper "runtime-report(calc.inclusive=true),profile.mpi" -i input.py

Build documentation
-------------------

Install the required Python packages to the virtual environment using ``pip``:

.. code-block:: shell

   cd doc
   pip install -r requirements.txt

.. important::

   Compiling documentation requires the **Python module of OpenSn**.

Then, from your ``build`` directory, generate the documentation with:

.. code-block:: shell

   cd doc
   make html

Once the build process is complete, you can view the generated documentation by
opening ``doc/html/index.html`` in your preferred web browser.
