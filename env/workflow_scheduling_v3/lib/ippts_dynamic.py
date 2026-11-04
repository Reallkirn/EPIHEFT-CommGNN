"""Independent IPPTS adaptation for the dynamic shared-channel environment.

The upstream IPPTS algorithm (Djigal et al., IEEE TPDS 2021) uses an
optimistic processor-cost table and a one-step successor look-ahead.  This
adapter preserves that processor-selection idea while replacing static
communication/processor availability with the current simulator interfaces.
It does not use HEFT_comm, learned features, or a learned policy.
"""
from __future__ import annotations
import math
import numpy as np


def _edge_cost(env, src_vm: int, dst_vm: int, data_size: float, now: float) -> float:
    """Cost of one future edge without mutating transfer state."""
    if data_size <= 0 or src_vm == dst_vm or env.comm_model is None:
        return 0.0
    cm = env.comm_model
    if getattr(cm, "channel_contention", False):
        if getattr(env, "_hb_interval", 0) > 0:
            return float(env._comm_cost_with_cached_load(
                src_vm, dst_vm, data_size, now))
        return float(cm.get_communication_cost_congested(
            src_vm, dst_vm, data_size, now))
    return float(cm.get_communication_cost(src_vm, dst_vm, data_size))


def _successor_lookahead(env, task, task_vm: int, task_eft: float,
                         workflow, now: float) -> float:
    """IPPTS-style optimistic one-step successor cost.

    For each direct successor, choose the best next VM using the current
    computation queue and a single parent edge.  Taking the maximum over
    successors retains the critical-successor behavior of the IPPTS PCM term.
    The estimate is optimistic: it does not reserve future transfers.
    """
    try:
        successors = list(workflow.app.successors(task))
    except Exception:
        successors = list(workflow.get_allnextTask(task))
    if not successors:
        return 0.0

    tail = 0.0
    for child in successors:
        data_size = float(workflow.get_data_size(task, child))
        child_best = math.inf
        for child_vm_idx, child_vm in enumerate(env.vm_queues):
            comm = _edge_cost(env, task_vm, child_vm_idx, data_size, now)
            child_ready = task_eft + comm
            child_exec = float(child_vm.get_taskExecuteTime(child, workflow))
            child_eft = max(float(child_vm.vmLatestTime()), child_ready) + child_exec
            child_best = min(child_best, child_eft)
        if math.isfinite(child_best):
            tail = max(tail, child_best - task_eft)
    return max(0.0, float(tail))


def ippts_dyn_action(env) -> int:
    """Return an IPPTS-Dyn processor choice for env.nextTask.

    The environment keeps its event-driven ready-task order.  IPPTS's
    independent processor-selection rule is applied to that current task:
    dynamic communication-ready time + processor queue + successor
    optimistic look-ahead.
    """
    if not env.vm_queues:
        return 0
    task = env.nextTask
    workflow = env.nextWrf
    now = float(env.nextTimeStep)
    scores = []
    for vm_idx, vm in enumerate(env.vm_queues):
        exec_time = float(vm.get_taskExecuteTime(task, workflow))
        if getattr(env, "comm_aware", False) and env.comm_model is not None:
            comm_time = float(env._observed_comm_time(
                task, vm_idx, workflow, now))
        else:
            comm_time = 0.0
        ready = now + comm_time
        eft = max(float(vm.vmLatestTime()), ready) + exec_time
        tail = _successor_lookahead(env, task, vm_idx, eft, workflow, now)
        scores.append(eft + tail)
    best = float(np.min(scores))
    candidates = np.flatnonzero(np.isclose(np.asarray(scores), best, rtol=1e-10, atol=1e-10))
    return int(candidates[0])
