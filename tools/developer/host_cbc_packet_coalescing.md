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
angle set finished. Pending sends are polled at entry; productive visits also
poll/flush at exit.

There is no new byte threshold, cell-count interval, timer, rank-count selection,
or prediction of critical messages. The pre-existing message-size limit remains
unchanged. This makes the batching decision structural, not a performance
heuristic; it does not make its performance independent of the message-size
configuration or the target MPI/fabric.

## Completed angle-set annotations

The host CBC FIFO scheduler can revisit already-finished sets while waiting for
other sets. CBC returns immediately on those visits, before entering the
`AngleSetAdvance` Caliper scope. This avoids annotation construction/destruction
on a no-work path, including its cost in Native runs with recording disabled.
No scheduler order, MPI progress, completion check, delayed-data drain, or
allocation policy changes. The scope still covers every unfinished advance,
including unsuccessful receive polling.

September 2026 Dane Caliper profiles showed that roughly half the host CBC
angle-set advances visited already-finished sets. New `AngleSetAdvance` visit
counts exclude those no-work visits; compare actual MPI calls and baseline
timings, not raw annotation counts across revisions. Lower annotation overhead
is not evidence of lower network latency or an improved wavefront critical path.

## Idle send-completion polling

An unfinished advance polls pending normal sends after receiving data. When no
cell task is ready, no sweep chunk runs and no new packet can be created during
that visit. In this case, do not immediately repeat the same send-completion poll
at exit. If sends remain pending, the next advance still polls them. If the
entry poll completed all sends, the existing completion check can finish the
angle set in this visit. Visits that sweep any cells still flush all partial
packets at exit, even if the last cell did not close a packet.

This changes polling frequency, not the completion condition, buffer lifetime,
task dependencies, or delayed-message drain. It needs no background MPI thread,
timer, or minimum-work threshold. Fewer completion polls are not automatically
faster: latency and progress depend on the MPI implementation and fabric. Keep
this experimental policy separate from the annotation-only change when measuring
on Dane, and include rendezvous traffic and cyclic/repeated sweeps in validation.

## Correctness and resource invariants

- Task-selection rules, dependency decrements, face-record format and numerical
  arithmetic do not change. Consecutive records share MPI envelopes. As before,
  actual task readiness/order can depend on MPI arrivals; the dependency partial
  order, not one particular total execution order, must be preserved.
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
