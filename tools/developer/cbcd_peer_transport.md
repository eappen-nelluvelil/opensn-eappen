# CBCD peer-transport experiment

## Scope and decision

This branch starts at baseline revision `547ef7bc80f4c3456c46cbf9549701c2b24b4c63`.
It tests a peer-based transport with one MPI progress thread per rank. It does
not import the fused kernels from the profiling-2 branch. The production,
memory-profiling, and baseline branches remain unchanged.

Keep this as an experiment pending Tuolumne measurements. Local median sweep
times are essentially unchanged. Some samples are faster but more variable.
There is no demonstrated improvement in timing stability, strong-scaling
efficiency, or weak-scaling efficiency yet. A monotonically decreasing grind
time is a measurement goal, not a property guaranteed by this protocol.

## Evidence motivating the experiment

The transferred `ee2d5d0a8` review and `e18993266` batched campaigns show frequent
MPI probes, many small messages, and substantial host-side communication work.
The prior analysis recorded approximately 2.52 million maximum-rank probes at
two nodes and 3.56 million at sixteen nodes for the review implementation.
The fused campaign is faster, but changes more than the communicator, so it
does not isolate the benefit of posted receives. Those campaigns also do not
establish weak-scaling gains for this prototype.

The previous changes that cache peer translations and traverse queue spans
are retained. The present experiment replaces repeated probing, blocking
receives, individual send tests, and per-packet send allocations. It does not
change the slot planner, sweep kernel, cell dependencies, reflecting-boundary
dependencies, or the equations solved.

## Transport and storage

`CBCDPeerTransport` owns two send buffers per destination and two receive
buffers per source. Nonblocking receives specify the source rank. One
`MPI_Testsome` checks both receive and send requests. A completed send releases
its buffer. A completed receive publishes sections to the existing angle-set
mailboxes. Workers still perform the final nonlocal FLUDS writes.

For a directed peer connection, let `T` be the bytes needed for all its face
records in one packet, including the packet header and one header per nonempty
angle-set section. Let `R` be the largest face record, and `H` the packet header
plus one section header. The buffer capacity is

```text
C = min(T, max(1 MiB, H + R))
```

The face record contains a destination face index, a value count, and its
double-precision flux values. Sender and receiver derive the same capacity
from their corresponding nonlocal faces, groups, and angles. No global
maximum or additional collective is needed. Zero-face peers post no receive.
An individual face that exceeds the MPI `int` count limit is rejected.

The application packet storage per rank is

```text
2 * sum(incoming peer capacities) + 2 * sum(outgoing peer capacities).
```

There are `2 * (number of sources + number of destinations)` request slots.
Storage depends on local peer degree rather than all MPI ranks. SPSC face
queues, mailboxes, FLUDS, and serialization metadata are additional storage.
The existing receive-pool type is not used by this transport.

Each progress pass assembles at most one packet per destination. It rotates
the first worker inspected for that destination and sends whatever currently
fits. There is no timer or minimum fill threshold. Records left in a worker
queue remain available to the next pass. Queue slots are returned only after
their referenced flux data have been serialized.

The window and 1 MiB target are experimental constants, not established
optima. Smaller packets can increase message count and alter which cells become
ready together. Larger windows consume more memory and may increase MPI
pressure. Ordinary `MPI_Isend` is retained. Therefore this bounds application
buffers and requests, not MPI's internal eager or unexpected-message storage.

## Ownership and progress invariants

1. Only the progress thread posts, tests, or cancels transport requests.
2. MPI owns a send buffer until completion. Its vector never moves or resizes
   while a request is active.
3. A completed receive starts with one producer reference plus one reference
   per published section. The producer releases its reference after publishing.
   Each worker releases its section after the FLUDS write.
4. Releases are atomic read-modify-write operations with release ordering. The
   progress thread uses an acquire load of zero before reposting. The release
   sequence makes every reader's completed access precede reuse.
5. Every incoming face is counted once per angle set and sweep. When that
   source's count reaches zero, all completions from the current bulk test are
   dispatched before any spare receives from that source are canceled and
   waited on. This avoids canceling a request that has already completed in
   the same bulk test but has not yet been dispatched.
6. The existing sweep barrier separates successive epochs. All outgoing
   queues and sends are drained, all source counts are exhausted, and workers
   have released their sections before the next sweep starts.
7. Workers visit all their active angle sets. Packet consumption must not
   depend on the readiness of a sweep kernel or reflecting predecessor.

The last condition requires a small change to angle-set initialization.
Workers prepare cell counters once and can place nonlocal flux while an angle
set waits on its reflecting predecessors. Boundary loading and kernel launches
still wait for those predecessors. Nonlocal and boundary flux use distinct
storage, so boundary loading does not overwrite these received values.
Without this early drain, two retained receive packets could prevent unrelated
traffic from the same peer from reaching an angle set needed to release the
reflection dependency. An unbounded packet pool previously concealed this
transport-induced wait cycle.

These rules follow the MPI buffer ownership and progress requirements:
[nonblocking communication](https://www.mpi-forum.org/docs/mpi-5.0/mpi50-report/node73.htm)
and [nonblocking semantics](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node75.htm).
They assume the existing acyclic device sweep graph and valid, matching
nonlocal-face metadata. This experiment does not add inter-rank cyclic sweeps
to device CBC.

## Local measurements, 2026-09-18

The comparison used a Native CUDA build with Clang 19, CUDA architecture 120,
Open MPI 5.0.10, and the same dependency/Python environment for both binaries.
The device was an NVIDIA GeForce RTX 5080 with driver 610.57.04.
Both cases ran on the same local GPU, with TCP host-buffer MPI transport and
four requested OpenSn threads per rank. This is not a multi-node or MI300A
measurement. Five fresh-process pairs were run at each rank count, alternating
which binary ran first. There were no simultaneous benchmark jobs.

The timing case has 12 cubed orthogonal cells, 128 directions, 64 groups,
single-angle aggregation, P0 scattering, and an eight-iteration Richardson
limit. It is a fixed-work performance case, not a converged accuracy benchmark.
The comparison checks every local scalar-flux value, not only extrema.
Maximum observed absolute field difference was `7.771561172376096e-16`.

| Ranks | Implementation | Median sweep (s) | Mean sweep (s) | Sample SD (s) |
| ---: | --- | ---: | ---: | ---: |
| 2 | Parent | 0.5508889 | 0.54998784 | 0.00231321 |
| 2 | Peer transport | 0.5497522 | 0.51078556 | 0.05382938 |
| 4 | Parent | 0.6077740 | 0.60722200 | 0.00176355 |
| 4 | Peer transport | 0.6066926 | 0.59941996 | 0.01722517 |

Median reductions of approximately 0.21% and 0.18% are too small to establish
a useful speedup here. The lower means reflect a few faster samples, with
substantially larger variation. Do not report the mean difference as a stable
performance improvement.

A separate four-rank, eight-cubed-cell run intercepted MPI calls over the full
application lifetime. Counts are sums over ranks, not critical-path times.

| MPI call | Parent | Peer transport |
| --- | ---: | ---: |
| `MPI_Iprobe` | 13,007,317 | 0 |
| `MPI_Recv` | 60,147 | 3 |
| `MPI_Test` | 60,282 | 0 |
| `MPI_Testsome` | 0 | 8,368,986 |
| `MPI_Isend` | 60,144 | 66,395 |

The remaining three blocking receives occur outside this transport. Probing is
gone, but idle bulk polling still calls MPI frequently. Send count increased by
about 10.4% in this example. Fewer calls or allocations alone are not proof of
lower runtime. The new packet cadence and window must be assessed together.

## Validation

- Native CUDA and CPU-only builds completed.
- Transport, packet, and queue unit tests passed at one, two, and four ranks
  in both builds. Coverage includes empty and one-way communication, zero-face
  sources, count limits, repeated sweeps, rendezvous-sized packets, a retained
  reader, two concurrent section readers, and occupied send windows.
- Focused GPU regressions passed for uniform reflecting transport, multiple
  groupsets with reflection and saved angular flux, an unstructured mesh, and
  Reed's problem with balance checks.
- The uniform reflecting problem has the independent constant solution
  `phi = q / Sigma_a = 1.25`, total absorption 64, and zero net leakage. It
  tests two executions for each of two separate problems. It passed with four
  requested threads and with `OPENSN_NUM_THREADS=1`. The latter's combined
  error metric was `2.216893335572e-12`, below the existing `1.0e-10` tolerance.
- Removing the early receive drain in a temporary build did not make that
  small reflecting regression fail. It is an accuracy and repeated-lifetime
  check, not a demonstrated reproducer of the transport-induced wait cycle.
  The safeguard follows from the ownership/dependency analysis above. A
  larger, deliberately imbalanced reflecting case remains a useful stress test.
- CUDA Compute Sanitizer memcheck, racecheck, initcheck, and synccheck passed
  on a two-rank case that constructs and executes two problems, including
  saved angular flux. This is focused coverage, not proof for all workloads.
- ThreadSanitizer passed the five peer-transport tests with two MPI ranks.
  The local PMIx shared-memory backend conflicted with sanitizer address
  mappings. Its recommended `PMIX_MCA_gds=hash` setting resolved that failure.
- AddressSanitizer and UndefinedBehaviorSanitizer passed the two-rank access
  tests with `OMPI_MCA_memory=^patcher` and leak detection disabled. Open MPI's
  allocation patcher crashed during initialization with ASan. LeakSanitizer
  subsequently reported MPI/PMIx/OPAL allocations even with zero tests selected.
  Those logs are retained. A clean whole-program leak check is not claimed.
- Host clang-tidy passed the new transport header and unit test with warnings
  treated as errors. CUDA clang-tidy is blocked by the installed CUDA header's
  `_NV_RSQRT_SPECIFIER` parse error. No repository-wide check was disabled.

The external debugging archive `cbcd-peer-transport-20260918` retains the logs,
benchmark and interception scripts, individual timing records, and saved fields.
The ordinary validation entry points are:

```sh
mpirun --np 4 BUILD/test/opensn-unit \
  --gtest_filter='CBCDPeerTransport.*:CBCDReceivePacket.*:LockFreeSPSCSlotQueue.*'
OPENSN_NUM_THREADS=4 test/run_tests --gpu --exe BUILD/python/opensn -j 16 -v 1 \
  -t test/python/modules/linear_boltzmann_solvers/transport_steady/transport_3d_uniform_reflecting_cbc_gpu.py
```

Repeat the unit selection with one and two ranks and both build configurations.
The other focused inputs are `transport_3d_reflecting_groupsets_cbc_gpu.py`,
`transport_3d_2_unstructured_cbc_gpu.py`, and `reed_balance_cbc_gpu.py` in the same
regression directory. No gold value or tolerance was changed.

## Tuolumne validation gate

Do not replace or modify a campaign currently collecting results. After it
finishes, use a separate campaign and source worktree for this branch, retaining
the parent campaign as the control. Reuse the dependency stack and venv, build
Native on an allocated node, and use the independent-process study framework
described in `cbcd_study.md`.

At 1, 2, 4, 8, and 16 nodes, retain four ranks per node, one device per rank,
21 requested threads per rank, and three fresh-process trials per mode. Test
both fixed-size strong scaling and the same weak-scaling meshes as the parent.
Collect baseline, Caliper+MPI, PMPI, and rank-zero ROCm traces. Keep incomplete
attempts and allocation provenance. Request allocations manually rather than
automatically chaining pdebug jobs.

Compare baseline medians, means, spread, actual unknown counts, normalized grind
times, strong-scaling efficiency, and work-normalized weak-scaling efficiency.
Check full residual histories and scalar-flux signatures before interpreting
timings. Inspect `ProgressPackets`, `MPI_Testsome`, send counts, packet sizes,
GPU launch cadence, and time waiting for incoming work. If memory growth
returns, use the separate lifetime diagnostics instead of inferring leaks from
grind-time variation.

Reject or refine the prototype if backpressure increases sweep times, reflection
stalls, numerical checks fail, or host memory rises materially. Test window or
packet-target changes one at a time. Do not add kernel fusion during this
comparison. No claim about Cray MPICH's internal locking is needed to retain
one transport MPI thread per rank.
