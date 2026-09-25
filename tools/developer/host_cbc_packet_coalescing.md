# Experimental host CBC packet coalescing

This branch tests a communication policy, not a different transport kernel or
discretization. Compare it to its parent branch, which already avoids redundant
normal receive probes and completion-only polling after interior cells.

## Policy

Normal face records for each peer use the existing packet builder and configured
`max_mpi_message_size`. When a queued record cannot fit in that peer's open packet,
the packet is closed and the angle set starts pending sends after the producing
cell. Otherwise records can accumulate while local tasks remain ready. At the
end of every angle-set visit, all remaining normal packets, including partial
ones, are started before waiting for additional dependencies or declaring the
angle set finished. Entry/exit completion polling remains active.

There is no new byte threshold, cell-count interval, timer, rank-count selection,
or prediction of critical messages. The pre-existing message-size limit remains
unchanged. This makes the batching decision structural, not a performance
heuristic; it does not make its performance independent of the message-size
configuration or the target MPI/fabric.

## Correctness and resource invariants

- Cell task order, dependency decrements, face records and numerical arithmetic
  do not change. Consecutive records are grouped into fewer MPI envelopes.
- Every face is copied into owned packet storage before its source flux storage
  can be reused. Packet buffers remain alive and immutable during active sends.
- Fragmented faces unlock their receiving cell only after all values arrive.
- Delayed payloads and completion markers retain the existing separate drain;
  sweep state is reset only after communication completion.
- No wait for a full packet occurs when local ready work is exhausted. Finite
  local work is followed by MPI progress and partial-packet flushing. No
  background asynchronous MPI progress is required for eventual progress.
- Packet data remain within the existing maximum size. More records can be
  retained before initiation, and reusable buffer capacities can consequently
  grow. This is not a claim of unchanged memory high-water marks.

## Performance hypothesis and acceptance

Coalescing reduces message startup/matching and request-management overhead.
It can also delay the downstream wavefront because the first record in a partial
packet is not sent immediately. In particular, latency-dominated strong scaling
may behave differently from a shared-memory benchmark. Lower message counts alone
are not evidence of a speedup.

Use paired fresh-process Native comparisons, identical mesh/quadrature/groups,
MPI binding, iteration count, compiler and packet limit. Compare residual/flux
signatures and report all timing samples. Keep Caliper/MPI and PMPI timings
separate from baseline performance. Measure strong and weak scaling on the target
fabric, including low local cell counts and memory usage, before promoting this
policy to the production branches.

The independent `communication_equilibrium.py` cases cover a reflecting
33-group absorber, complete/fragmented messages, two groupsets, cyclic partitions,
serial/MPI and repeated execution with balance. Also exercise the existing
nonuniform cyclic, restart, transient and balance regressions; repeat MPI tests
with rendezvous-sized messages. A converged answer alone is not a correctness
reference or a performance guarantee.
