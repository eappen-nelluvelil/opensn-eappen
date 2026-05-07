# Node-to-Node Scaling Study

This is a node-to-node scaling study for OpenSn. It currently generates tetrahedral unstructured-mesh
transport studies for the AAH or CBC sweep algorithms, with CPU execution by default and optional GPU
execution.

## 1. Generate inputs

``generate_scaling_study.py`` generates the meshes and launch scripts necessary for strong and weak scaling studies.
Edit ``strong_nodes`` and ``weak_nodes`` in the script to choose separate node counts for strong and
weak scaling. For example, CBC strong scaling can omit 1-node runs while weak scaling still includes
1 node.

Meshes are cached under ``output/meshes`` and reused across study directories that share the same
mesh configuration. This is useful when generating strong- or weak-scaling scripts for multiple
OpenSn binaries with the same ranks-per-node and node/divisor configuration: only the launch scripts
and input files are regenerated for each binary-specific study directory.

Usage:
```
python3 generate_scaling_study.py
  --type {strong,weak}  Type of scaling test to generate files for.
  --sweep-type {AAH,CBC}
                        Sweep algorithm to use in the generated OpenSn input.
                        Defaults to AAH.
  --use-gpus            Enable GPU sweep execution in the generated OpenSn input
                        and launch scripts.
  --engine {slurm,lc-flux,alcf-pbs}
                        Job submitting system. Defaults to slurm.
  --opensn-binary OPENSN_BINARY
                        Path to the OpenSn executable.
  --study-name STUDY_NAME
                        Optional identifier appended to output directories and
                        job names, for example a branch or binary label.
  --nodes NODES         Comma-separated node counts for this study.
  --cores-per-node CORES_PER_NODE
                        CPU cores per node.
  --gpus-per-node GPUS_PER_NODE
                        GPUs per node for GPU studies.
  --strong-divisor STRONG_DIVISOR
                        Gmsh divisor for the fixed strong-scaling mesh.
```

Weak-scaling mesh divisors are scaled from the configured reference divisors using the selected
ranks per node, so GPU studies with fewer ranks per node generate appropriately smaller weak-scaling
meshes while still reusing meshes across studies with the same node and rank configuration.

Examples:
```
python3 generate_scaling_study.py --type=strong --sweep-type=AAH --engine=slurm
```
```
python3 generate_scaling_study.py --type=strong --sweep-type=CBC --engine=slurm
```
```
python3 generate_scaling_study.py --type=strong --sweep-type=CBC --engine=slurm \
  --opensn-binary=/path/to/build/python/opensn \
  --study-name=cbc-cycles-2
```
```
python3 generate_scaling_study.py --type=weak --sweep-type=CBC --use-gpus --engine=lc-flux
```
```
python3 generate_scaling_study.py --type=weak --sweep-type=CBC --use-gpus --engine=lc-flux \
  --nodes=1,2,4,8,16,32,64,128,256 \
  --cores-per-node=96 --gpus-per-node=4
```

After running the script, a folder ``output/{strong/weak}_{aah/cbc}_{cpu/gpu}`` will appear. If
``--study-name`` is used, the folder is suffixed with that study name, e.g.
``output/strong_cbc_cpu_cbc-cycles-2``.
The generated input files reference the shared meshes in ``output/meshes``.
Change directory into that folder for the next step.

## 2. Submitting jobs

Inside of the output folder (``output/*/``), execute:
```
source submit_jobs.sh
```
and wait for all the jobs to finish.

When all jobs have been finished, go back the ``scaling`` folder (i.e. exit the ``output/`` folder).

## 3. Plot

To plot strong/weak scaling, run:
```
python3 plot_strong.py --sweep-type=AAH
```
```
python3 plot_weak.py --sweep-type=CBC --use-gpus
```

Custom result folders can be provided with ``--dir``:
```
python3 plot_strong.py --dir=output/strong_cbc_cpu --sweep-type=CBC
```
Multiple result folders can be overlaid by specifying ``--dir`` and ``--label`` more than once:
```
python3 plot_strong.py \
  --dir=output/strong_cbc_cpu_cbc-cycles-2 --label=cbc-cycles-2 \
  --dir=output/strong_cbc_cpu_cyclic-deps-stages --label=cyclic-deps-stages \
  --sweep-type=CBC
```

Result of the current run can be compared with or recorded with:
```
python plot_strong.py
  --history {none,comp,save}
                        History mode for the plot:
                            none (default, only plot current data),
                            comp (compare with history in the same plot without saving),
                            save (plot and overwrite current history value).
```
