# CUDA Cycle-Union Placement

This note decides where temporal cycle-union preprocessing runs for the CUDA
backend: on the host before launch, or on the GPU. It is the Phase 8.7 decision.

## Current state

The temporal CUDA dispatch already builds start events on the host and drops the
ones whose cycle-union set is empty before launching the kernel
(`cuda_work_item.hpp`, Phase 7.3). The prefilter is correctness preserving and
unit tested. It removes dead start edges, which both shrinks the launch and
improves load balance by not handing empty work to the device.

## Options

1. **Keep cycle-union on the host (chosen).** The preprocessing is a sparse
   double reachability search per start event. It overlaps naturally with the
   host-to-device transfer and kernel of the previous batch, and it reuses the
   already-validated sequential implementation. Its output, the surviving start
   events, is a compact array that streams to the device cheaply.

2. **Move cycle-union to the GPU.** A device cycle-union would run a parallel
   reachability pass per start event before the counting kernel. It only pays off
   if host preprocessing becomes a measurable fraction of end-to-end time, which
   requires either very many start events or a host that cannot overlap the work
   with the device.

## Decision

Keep cycle-union preprocessing on the host for now. There is no profiling
evidence that it is a bottleneck, the host implementation is already validated,
and host preprocessing overlaps with device execution. Porting it to the GPU adds
a second irregular reachability kernel and its own correctness surface for no
demonstrated benefit.

## Revisit criteria

Move cycle-union to the GPU only if cluster profiling shows all of:

- host preprocessing time is a significant and non-overlapped fraction of total
  runtime on the target datasets, and
- the start-event count is large enough that a parallel device reachability pass
  would clearly beat the host, and
- the counting kernel is already balanced, so preprocessing, not the search, is
  the limiting stage.

Until those hold, the host prefilter remains the better choice. The NVTX ranges
added for profiling separate transfer, preprocessing, and kernel time so this
fraction can be measured directly.
