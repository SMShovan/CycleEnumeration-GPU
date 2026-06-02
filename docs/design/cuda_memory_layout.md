# CUDA Memory Layout

This note describes the device memory layout used by the CUDA cycle kernels and
the reasoning behind it.

## Structure of arrays for the hot adjacency

The CPU `GraphView` stores outgoing adjacency as an array of `AdjacencyEntry`
structs, each holding a neighbor id, a logical edge id, and a timestamp range.
That array-of-structs layout is convenient on the CPU but wastes device memory
bandwidth: a kernel that only needs the neighbor id still pulls the edge id and
both timestamp offsets into the cache line, and adjacent threads read addresses
that are a full struct apart, so the loads do not coalesce.

`pack_graph_for_cuda` therefore also emits the hot outgoing adjacency as a
structure of arrays:

- `outgoing_neighbors` — one 32-bit `VertexId` per edge slot.
- `outgoing_timestamp_begin` / `outgoing_timestamp_end` — 64-bit device offsets
  per edge slot.

All three are indexed by the same CSR offset as the original
`outgoing_edges` array, so a kernel walks `outgoing_offsets[v] .. offsets[v+1]`
exactly as before, but each field lives in its own contiguous array. When a warp
reads consecutive edge slots, the loads of a single field are contiguous and
coalesce into a few transactions, and the static kernel, which only needs the
neighbor, never fetches timestamp data at all.

The array-of-structs `outgoing_edges` is retained for the incoming/CSC side and
for host code and tests that want a single entry, but the device traversal path
uses only the split arrays through the `DeviceGraphView`.

## Index widths and overflow

Vertex and edge ids stay 32-bit (`VertexId`, `EdgeId`) because single-node graphs
in scope stay well under 2^32 vertices, and the narrower ids halve the neighbor
array footprint and bandwidth. Offsets into the timestamp and adjacency arrays
use the 64-bit `DeviceOffset` type so timestamp-heavy temporal graphs remain
representable. Every host-to-device width narrowing goes through
`checked_offset`, which throws `std::overflow_error` rather than silently
truncating, so an out-of-range graph fails clearly at pack time instead of
corrupting device traversal.

## Upload path

`upload_graph` centralizes the device allocation and host-to-device copies for
the offsets, the three split adjacency arrays, and, for the time-window and
temporal paths, the flat timestamp array. It returns an owner object whose
`view()` produces the `DeviceGraphView` passed to kernels. Centralizing the
upload keeps the static, time-window, and temporal entry points consistent and
gives one place to measure and tune transfer time.

## Measurement

The split layout is expected to reduce kernel time on neighbor-bound traversal
and to reduce transfer volume for the static path. Transfer time and kernel time
before and after the change are recorded on the H100 cluster through the
benchmark harness; the host packing itself is covered by unit tests that assert
the split arrays mirror the array-of-structs layout entry by entry.
