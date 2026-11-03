"""Discrete-event simulator for asynchronous, reactive tuning on a replayed learning-curve pool.

Semantics (kept deliberately simple; every choice here is an assumption to defend in the chapter):
  * K workers train one configuration each. After each epoch the run REPORTS its metric.
  * A policy sees each report and returns CONTINUE or STOP, plus the decision latency L (epoch units).
  * Training does not pause while a decision is computed. A STOP therefore takes effect L after the
    report: epochs finished meanwhile are wasted compute, and the worker is only freed at report+L.
    A CONTINUE costs nothing, so latency only hurts when the answer is STOP (or when proposing).
  * When a worker frees, the proposer picks the next configuration; proposing costs `propose_latency`
    during which the worker idles.
  * Quality score at time t = best FINAL metric among runs that completed all epochs by t
    (the value you could actually deploy). Runs stopped early never count.
"""
import heapq
from dataclasses import dataclass, field
import numpy as np

CONTINUE, STOP = 0, 1


@dataclass
class Run:
    cfg: int
    worker: int
    start: float
    epoch: int = 0                     # epochs reported so far
    stopped_at: float = None           # time the stop takes effect (None = not stopping)


@dataclass
class SimState:
    now: float
    pool: object
    runs: list                         # every Run started so far (index = run id)
    reports: dict                      # cfg -> list of metrics reported so far
    completed: dict                    # cfg -> final metric
    running: dict                      # worker -> run id
    done_cfgs: set = field(default_factory=set)   # started configs (never re-run)


def simulate(pool, proposer, policy, n_workers, t_max, rng, propose_latency=0.0, grid=None):
    """Returns dict with the anytime quality trace and accounting. `proposer(state, rng) -> cfg`,
    `policy(state, run, metric, rng) -> (action, latency)`."""
    st = SimState(0.0, pool, [], {}, {}, {}, set())
    heap, seq = [], 0

    def push(t, kind, payload):
        nonlocal seq
        heapq.heappush(heap, (t, seq, kind, payload)); seq += 1

    def start_run(w, t0):
        cfg = proposer(st, rng)
        st.done_cfgs.add(cfg)
        run = Run(cfg, w, t0)
        st.runs.append(run); rid = len(st.runs) - 1
        st.running[w] = rid; st.reports[cfg] = []
        push(t0 + pool.dt(cfg, 1), "epoch", rid)

    for w in range(n_workers):
        push(propose_latency, "free", (w, -1))
    best, trace, busy = -np.inf, [(0.0, -np.inf)], 0.0
    E = pool.n_epochs
    while heap:
        t, _, kind, payload = heapq.heappop(heap)
        if t > t_max:
            break
        st.now = t
        if kind == "free":                                   # a worker (re)starts, or a stop lands
            w, rid = payload
            if rid >= 0:
                run = st.runs[rid]
                if st.running.get(w) != rid:                 # run already completed before stop landed
                    continue
                busy += t - run.start
                del st.running[w]
                push(t + propose_latency, "free", (w, -1)); continue
            start_run(w, t)
            continue
        run = st.runs[payload]                               # an epoch finishes
        w = run.worker
        if st.running.get(w) != payload or (run.stopped_at is not None and t >= run.stopped_at):
            continue                                         # run was stopped; this epoch never reports
        run.epoch += 1
        m = float(pool.y[run.cfg, run.epoch - 1])
        st.reports[run.cfg].append(m)
        if run.epoch == E:                                   # completed
            st.completed[run.cfg] = m
            busy += t - run.start
            del st.running[w]
            if m > best:
                best = m; trace.append((t, best))
            push(t + propose_latency, "free", (w, -1)); continue
        if run.stopped_at is None:
            action, lat = policy(st, run, m, rng)
            if action == STOP:
                run.stopped_at = t + lat
                push(t + lat, "free", (w, payload))
        push(t + pool.dt(run.cfg, run.epoch + 1), "epoch", payload)
    for w, rid in st.running.items():                        # account for runs still in flight
        busy += t_max - st.runs[rid].start
    grid = np.linspace(0, t_max, 101) if grid is None else grid
    ts = np.array([x[0] for x in trace]); bs = np.array([x[1] for x in trace])
    q = np.array([bs[np.searchsorted(ts, g, side="right") - 1] for g in grid])
    return dict(grid=grid, quality=q, n_started=len(st.runs), n_completed=len(st.completed),
                worker_time=busy, final=q[-1])
