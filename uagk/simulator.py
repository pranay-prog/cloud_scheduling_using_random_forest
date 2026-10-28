"""Event-driven cloud cluster simulator (heap-based discrete-event engine).

Model
  * Arrivals go to the processing queue Q (capacity Q_max); when Q is full
    they wait in the ingress buffer B (capacity B_max) and move to Q in FCFS
    order as space frees. Arrivals finding both full are dropped (counted as
    SLA violations).
  * After service each result is written to storage (T_storage per request,
    not occupying a worker).
  * End-to-end latency  W_total = W_B + W_q + S + T_storage.
  * k(t) identical workers. Scale-up starts VMs that become usable after
    T_scale (billed while booting). Scale-down cancels booting VMs first,
    then removes idle workers, then retires busy workers when their job ends.
  * A controller calls policy.decide(obs) every T_ctrl seconds.
Metrics are computed for jobs arriving in [warmup, horizon]; arrivals stop at
the horizon and the cluster drains so every measured job completes.
"""
from __future__ import annotations

import bisect
import heapq
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .workload import Jobs
from .oracle import Prediction

ARR, DONE, BOOT, CTRL, MARK = 0, 1, 2, 3, 4


@dataclass
class SimConfig:
    horizon: float = 2400.0
    warmup: float = 300.0
    t_ctrl: float = 4.0
    t_scale: float = 10.0
    t_sla: float = 4.0
    q_max: int = 200              # processing-queue capacity Q_max
    b_max: int = 3000             # ingress-buffer capacity B_max
    k_min: int = 2
    k_max: int = 120
    k0: int = 20
    window: float = 30.0          # characterisation window (seconds)


@dataclass
class Obs:
    """Everything a policy may look at when deciding (step 1 telemetry)."""
    t: float
    cfg: SimConfig
    arr_times: np.ndarray        # arrivals in window
    pred_idx: slice              # indices of those jobs (for RF predictions)
    done_s: np.ndarray           # actual runtimes of jobs completed in window
    t_store: np.ndarray          # storage-write times of jobs in window (known from size)
    queue_len: int               # B(t) + Q(t)
    buf_len: int                 # B(t)
    busy: int
    active: int
    booting: int
    retiring: int
    util: float                  # time-averaged busy fraction over last interval


@dataclass
class Result:
    resp: np.ndarray             # W_total
    wait: np.ndarray             # W_B + W_q
    wait_b: np.ndarray
    wait_q: np.ndarray
    service: np.ndarray
    storage: np.ndarray
    dropped: np.ndarray
    vm_seconds: float
    k_trace: np.ndarray          # (t, active, booting) samples per control step
    n_scale_actions: int
    measured_span: float


class Cluster:
    def __init__(self, jobs: Jobs, pred: Prediction | None, policy, cfg: SimConfig):
        self.j, self.pred, self.policy, self.cfg = jobs, pred, policy, cfg
        n = len(jobs)
        self.start = np.full(n, np.nan)
        self.end = np.full(n, np.nan)
        self.dropped = np.zeros(n, bool)
        self.enter_q = np.full(n, np.nan)
        self.t_store = jobs.t_store if getattr(jobs, "t_store", None) is not None else np.zeros(n)
        self.qp = deque()        # processing queue Q(t)
        self.buf = deque()       # ingress buffer B(t)
        self.ev = []
        self.seq = 0
        self.now = 0.0
        self.active = cfg.k0
        self.booting = 0
        self.retire_pending = 0
        self.busy = 0
        self.boot_live = {}      # boot id -> alive flag
        self.boot_counter = 0
        self.vm_time = 0.0
        self.busy_time = 0.0
        self.last = 0.0
        self.done_t, self.done_s = [], []
        self.marks = {}
        self.k_trace = []
        self.actions = 0
        self._last_busy_time = 0.0
        self._last_active_time = 0.0
        self.active_time = 0.0

    # ------------------------------------------------------------ engine
    def push(self, t, kind, payload=0):
        self.seq += 1
        heapq.heappush(self.ev, (t, self.seq, kind, payload))

    def tick(self):
        dt = self.now - self.last
        if dt > 0:
            self.vm_time += (self.active + self.booting) * dt
            self.busy_time += self.busy * dt
            self.active_time += self.active * dt
            self.last = self.now

    def run(self) -> Result:
        cfg = self.cfg
        if len(self.j):
            self.push(self.j.t_arr[0], ARR, 0)
        self.push(cfg.t_ctrl, CTRL)
        self.push(cfg.warmup, MARK, 0)
        self.push(cfg.horizon, MARK, 1)
        n = len(self.j)
        while self.ev:
            t, _, kind, p = heapq.heappop(self.ev)
            self.now = t
            self.tick()
            if kind == ARR:
                self._arrive(p)
                if p + 1 < n:
                    self.push(self.j.t_arr[p + 1], ARR, p + 1)
            elif kind == DONE:
                self._done(p)
            elif kind == BOOT:
                self._boot_done(p)
            elif kind == CTRL:
                self._control()
                if t + cfg.t_ctrl < cfg.horizon or self.qp or self.buf or self.busy:
                    self.push(t + cfg.t_ctrl, CTRL)
            elif kind == MARK:
                self.marks[p] = self.vm_time
        lo = np.searchsorted(self.j.t_arr, cfg.warmup)
        hi = np.searchsorted(self.j.t_arr, cfg.horizon)
        sl = slice(lo, hi)
        arr = self.j.t_arr[sl]
        wait = self.start[sl] - arr
        wait_b = self.enter_q[sl] - arr
        wait_q = self.start[sl] - self.enter_q[sl]
        service = self.end[sl] - self.start[sl]
        storage = self.t_store[sl]
        resp = self.end[sl] + storage - arr
        return Result(resp, wait, wait_b, wait_q, service, storage,
                      self.dropped[sl], self.marks[1] - self.marks[0],
                      np.array(self.k_trace), self.actions, cfg.horizon - cfg.warmup)

    # ------------------------------------------------------------ jobs
    def _arrive(self, i):
        if self.active - self.busy > 0 and not self.qp and not self.buf:
            self.enter_q[i] = self.now
            self._start(i)
        elif len(self.qp) < self.cfg.q_max and not self.buf:
            self.enter_q[i] = self.now
            self.qp.append(i)
        elif len(self.buf) < self.cfg.b_max:
            self.buf.append(i)
        else:
            self.dropped[i] = True

    def _start(self, i):
        self.busy += 1
        self.start[i] = self.now
        self.push(self.now + self.j.s_true[i], DONE, i)

    def _done(self, i):
        self.busy -= 1
        self.end[i] = self.now
        self.done_t.append(self.now)
        self.done_s.append(self.j.s_true[i])
        if self.retire_pending > 0:
            self.retire_pending -= 1
            self.active -= 1
        self._dispatch()

    def _refill(self):
        while self.buf and len(self.qp) < self.cfg.q_max:
            i = self.buf.popleft()
            self.enter_q[i] = self.now
            self.qp.append(i)

    def _dispatch(self):
        while self.active - self.busy > 0 and (self.qp or self.buf):
            if not self.qp:
                self._refill()
            self._start(self.qp.popleft())
            self._refill()

    # ------------------------------------------------------------ scaling
    def _boot_done(self, bid):
        if self.boot_live.pop(bid, False):
            self.booting -= 1
            self.active += 1
            self._dispatch()

    def scale_to(self, target):
        cfg = self.cfg
        target = int(min(max(target, cfg.k_min), cfg.k_max))
        current = self.active + self.booting - self.retire_pending
        if target == current:
            return
        self.actions += 1
        if target > current:
            diff = target - current
            undo = min(diff, self.retire_pending)
            self.retire_pending -= undo
            diff -= undo
            for _ in range(diff):
                self.boot_counter += 1
                self.boot_live[self.boot_counter] = True
                self.booting += 1
                self.push(self.now + cfg.t_scale, BOOT, self.boot_counter)
        else:
            diff = current - target
            for bid in sorted(self.boot_live, reverse=True):
                if diff == 0:
                    break
                if self.boot_live[bid]:
                    self.boot_live[bid] = False
                    self.booting -= 1
                    diff -= 1
            idle = self.active - self.busy
            take = min(diff, idle)
            self.active -= take
            diff -= take
            self.retire_pending += diff

    # ------------------------------------------------------------ control
    def _control(self):
        cfg = self.cfg
        t = self.now
        lo = np.searchsorted(self.j.t_arr, t - cfg.window)
        hi = np.searchsorted(self.j.t_arr, t, side="right")
        d_lo = bisect.bisect_left(self.done_t, t - cfg.window)
        d_busy = self.busy_time - self._last_busy_time
        d_act = self.active_time - self._last_active_time
        self._last_busy_time, self._last_active_time = self.busy_time, self.active_time
        util = d_busy / d_act if d_act > 0 else 1.0
        obs = Obs(t, cfg, self.j.t_arr[lo:hi], slice(lo, hi),
                  np.asarray(self.done_s[d_lo:]), self.t_store[lo:hi],
                  len(self.qp) + len(self.buf), len(self.buf), self.busy,
                  self.active, self.booting, self.retire_pending, util)
        target = self.policy.decide(obs, self.pred)
        if t < cfg.horizon:
            self.scale_to(target)
        self.k_trace.append((t, self.active, self.booting))


def summarize(res: Result, t_sla: float) -> dict:
    ok = ~res.dropped & np.isfinite(res.resp)
    resp = np.where(ok, res.resp, np.inf)
    viol = np.mean(resp > t_sla)
    finite = resp[np.isfinite(resp)]
    m = lambda x: float(np.nanmean(np.where(ok, x, np.nan)))
    wait_ok = np.where(ok, res.wait, 0.0)
    return dict(
        sla_violation=float(viol),
        p95=float(np.percentile(finite, 95)) if len(finite) else np.nan,
        p99=float(np.percentile(finite, 99)) if len(finite) else np.nan,
        mean_resp=float(finite.mean()) if len(finite) else np.nan,
        mean_wait=m(res.wait),
        mean_wb=m(res.wait_b), mean_wq=m(res.wait_q),
        mean_service=m(res.service), mean_storage=m(res.storage),
        p95_wait=float(np.percentile(wait_ok[ok], 95)) if ok.any() else np.nan,
        total_wait_hours=float(wait_ok.sum() / 3600.0),
        drop_rate=float(res.dropped.mean()),
        vm_hours=res.vm_seconds / 3600.0,
        mean_k=res.vm_seconds / res.measured_span,
        scale_actions=res.n_scale_actions,
        n_jobs=int(len(resp)),
    )
