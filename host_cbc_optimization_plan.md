# Host CBC cyclic-dependency optimization plan

This note records concrete optimization opportunities for the refactored host
cell-by-cell (CBC) sweep algorithm after adding support for cyclic dependencies
and lagged angular fluxes. The emphasis is preserving the current safety
properties:

- A cell task may execute only after all non-delayed incoming dependencies are
  satisfied.
- Delayed local and nonlocal dependencies use lagged angular fluxes from the
  previous fixed-point iterate.
- Incomplete nonlocal payloads must never make a downstream task ready.
- Payload-size and slot mismatches should fail deterministically rather than
  hanging at runtime.

The current implementation already supports cyclic local and MPI dependencies,
metadata validation, cached delayed-location lookups, cached receive-source
routing, and direct partial reassembly into FLUDS-owned receive buffers. The
remaining opportunities below are organized by risk and implementation scope.

## Highest-value safe items

### 1. Cache CBC task state across sweeps

Current CBC angle-set advancement initializes dependency counts, completion
flags, and the initial ready-task stack lazily at the start of each sweep. These
data are derived from the immutable `CBC_SPDS::GetTaskList()` and do not change
during repeated sweeps for the same angle set.

The optimization is to precompute:

- Initial dependency counts for every task.
- Initial ready tasks.
- Scratch completion flags and remaining dependency counts sized to the task
  list.

Each sweep then resets mutable state with `std::copy`, `std::fill`, and an
assignment of the initial ready-task stack. This mirrors the older cyclic CBC
branch and the general AAH pattern of keeping immutable schedule/message
metadata separate from per-sweep mutable state. It is safe because the task graph
is still owned by SPDS and the reset only restores the same initial state that
the lazy setup currently constructs.

Expected benefit: lower per-sweep overhead, especially for problems with many
source iterations or Krylov applications where the same angle sets are advanced
many times.

Natural commit boundary: one commit touching `CBC_AngleSet`.

### 2. Track immediate receive completion per source rank

Current CBC immediate receive polling scans every immediate predecessor source
rank on every `ReceiveData()` call. After a predecessor has delivered all
non-delayed face payloads for the current sweep, probing that source again is
unnecessary.

The optimization is to precompute the expected number of immediate incoming face
payloads per source rank during FLUDS/communicator setup. During a sweep, the
communicator increments a received count whenever a complete immediate face
payload is committed. Once the count reaches the expected count, that source is
marked complete and is no longer probed until communicator reset.

This is analogous to AAH's receive flags for message blocks, but adapted to
CBC's face-level task-unlock semantics. It preserves CBC's fine-grained
readiness because each face payload is still committed independently.

Expected benefit: less MPI probing in problems with many predecessor ranks or
many scheduler passes while waiting for a small subset of active sources.

Natural commit boundary: one commit adding source metadata to
`CBC_FLUDSCommonData`/`CBC_AsynchronousCommunicator`.

### 3. Use adaptive CBC payload chunk sizing

The current CBC communicator caps immediate message payloads with a conservative
hard limit. This bounds message size, but it can split one face payload into many
chunks. A cell cannot be unlocked until all chunks for an incoming face have
arrived, so excessive chunking typically increases header traffic, MPI message
count, and duplicate-chunk bookkeeping without improving readiness.

The optimization is to derive CBC chunk size from the configured MPI message
limit, with only a safety floor for the message header. This keeps the existing
bounded-message guarantee while respecting user/runtime configuration.

Expected benefit: fewer chunks for high-group-count cases and fewer partial
payload bookkeeping operations. This should especially help lagged or immediate
payloads with many groups.

Natural commit boundary: one communicator-only commit.

### 4. Persist reusable communicator buffers across sweeps

The communicator currently clears reusable send buffers on reset. That discards
capacity learned during previous sweeps and can reintroduce allocator traffic in
every source iteration.

The optimization is to preserve reusable buffer capacity across `Reset()` while
clearing only in-flight state. A bounded retention policy should be used so that
pathological one-time large messages do not permanently inflate memory use. The
safe minimal policy is to retain buffers whose capacity is no larger than the
configured message bound.

Expected benefit: lower allocator overhead during repeated sweeps with similar
communication structure.

Natural commit boundary: one communicator-only commit.

### 5. Preallocate partial chunk tracking

For chunked messages, `PartialIncomingPayload` currently initializes its
received-chunk flag vector when the first chunk of a payload arrives. The number
of chunks is known from the expected payload size and the communicator chunk
size.

The optimization is to preallocate the received-chunk flags for every immediate
and delayed face slot during communicator construction or reset. A generation or
active flag can distinguish inactive partial payloads from active incomplete
payloads. This removes allocation from the receive hot path while keeping
duplicate-chunk detection.

Expected benefit: lower receive-side allocator overhead in high-group-count or
small-message-limit cases.

Natural commit boundary: one communicator-only commit.

## AAH-inspired CBC-specific improvements

### 6. Precompute a CBC message plan analogous to AAH message data

AAH builds message structures once and then uses those structures for direct MPI
send/receive blocks. CBC cannot simply aggregate all immediate data by
predecessor because cell readiness is face-granular: a downstream cell should be
able to execute as soon as its last incoming face arrives, not after all faces
from an upstream rank arrive.

The CBC analog should therefore precompute:

- Face-slot headers or packed header templates.
- Source rank and destination peer for every face slot.
- Expected payload sizes and chunk counts.
- Per-peer reserve sizes for send buffers.
- Per-source expected immediate payload counts.

The immediate path would remain face-granular, while the metadata and buffer
sizing would be precomputed.

Expected benefit: less repeated metadata lookup, fewer branches in sweep
kernels, better send-buffer reservation, and easier validation.

Risk: low to medium, depending on how much metadata is moved out of the sweep
kernel path.

### 7. Use direct receive into FLUDS where CBC semantics allow it

AAH receives directly into FLUDS-owned blocks. CBC currently receives packed byte
messages and copies payload chunks into FLUDS-owned buffers. Direct receive is
hard for immediate multi-record messages because one MPI receive can contain
several face records with headers. However, delayed nonlocal payloads do not
unlock work during the current sweep, so delayed communication is a better
candidate for direct block receive.

Expected benefit: fewer copies and simpler receive-side parsing for delayed
payloads.

Risk: medium. The design must preserve delayed completion markers or replace
them with an equivalent per-source completion protocol.

### 8. Aggregate delayed CBC payloads by delayed predecessor

Delayed nonlocal angular fluxes are consumed only in the next lagged iteration.
They do not need to unlock cell tasks during the current sweep. Therefore, they
can be communicated in larger predecessor-indexed blocks, similar to AAH delayed
messages, while immediate non-delayed CBC payloads remain face-granular.

Expected benefit: substantially fewer delayed MPI records and less delayed
receive parsing.

Risk: medium. The delayed FLUDS layout must provide stable block offsets, and
the delayed completion protocol must remain deterministic.

### 9. Precompute per-local-face sweep metadata

The sweep kernels repeatedly query whether a face is local/nonlocal/delayed, the
incoming/outgoing slot, destination location, peer index, and face-node count.
Much of this is already available in `CBC_FLUDSCommonData`.

The optimization is to expose a compact per-local-face metadata table that the
sweep kernels can read once per face. This would reduce repeated accessor calls
and branch reconstruction in generic and fixed-node kernels.

Expected benefit: lower instruction overhead inside every swept cell,
particularly for unstructured meshes.

Risk: low to medium. It changes data layout but not sweep semantics.

## More creative or higher-risk ideas

### 10. Payload- and physics-aware feedback-arc weighting

CBC cyclic behavior depends on which graph edges are lagged. The current global
edge weights use angular upwind strength based on face area and directional
cosine. A stronger objective could include payload and computational cost:

`edge_weight ~ |omega . n| * area * face_dofs * groups * angles`

The feedback-arc solver would then prefer lagging weak or cheap dependencies,
potentially improving fixed-point convergence and communication cost.

Expected benefit: fewer iterations or cheaper lagged payloads on cyclic
partition graphs.

Risk: medium to high. It can change numerical iteration behavior and should be
validated across representative cyclic meshes.

### 11. Adaptive CBC send coalescing

CBC currently sends aggressively to maximize downstream task availability. This
is good for latency, but it can produce many small messages. A bounded adaptive
policy could flush when:

- The ready queue drains.
- A peer buffer reaches a byte threshold.
- A fixed number of cells has been swept.
- The scheduler detects no local ready work and needs communication progress.

Expected benefit: fewer small MPI messages and better bandwidth utilization.

Risk: medium to high. Excessive batching can destroy CBC's early-unlock
advantage, so this requires profiling.

### 12. MPI_ANY_SOURCE receive polling with strict validation

Instead of scanning every possible source rank, CBC could probe with
`MPI_ANY_SOURCE` for the angle-set tag and validate the returned source against a
precomputed allowed-rank table. This may reduce probe overhead when only a few
sources are active.

Expected benefit: less polling overhead with many possible predecessors.

Risk: medium. MPI implementation behavior varies, and fairness/order properties
must be checked carefully.

### 13. Ready-task ordering by SPDS/topological order

CBC currently uses a LIFO ready-task stack. A deterministic ordering based on
SPDS/topological position may improve cache locality and communication
regularity. The older cyclic branch included related machinery for cylindrical
ordering.

Expected benefit: potentially better cache behavior and smoother downstream
communication.

Risk: low to medium. It may improve or hurt depending on mesh and partitioning,
so it should be benchmarked.

## Implementation sequence for this branch

The first implementation pass should focus on the highest-value safe items:

1. Cache CBC task-state reset metadata.
2. Track immediate receive completion per source rank.
3. Use adaptive configured chunk sizing.
4. Persist bounded reusable communicator buffers across sweeps.
5. Preallocate partial chunk tracking.

Each item should be committed separately. The AAH-inspired and higher-risk
ideas should remain documented until the low-risk set has been verified and
profiled, because they either change delayed communication layout or can alter
the latency/bandwidth balance of CBC.
