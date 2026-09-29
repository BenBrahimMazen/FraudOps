"""Monitor window fetch: the labelled lookback must clear the label delay.

Regression: with window_sim_days == label_delay_days the trailing predictions
window is NEVER fully labelled (a prediction becomes labelable exactly
``delay`` days after scoring), so the labelled-performance lookup used to
fetch an empty set and performance monitoring silently never ran. This test
pins the lookback arithmetic against a scripted connection.
"""

from __future__ import annotations

from fraudops.monitoring.monitor import fetch_window

DAY = 86_400
MAX_DT = 182 * DAY


class ScriptedCursor:
    """Returns canned results in execute order (max sim_ts, preds, feats, labelled)."""

    def __init__(self, results: list) -> None:
        self._results = results
        self.queries: list[tuple[str, tuple | None]] = []
        self.executemany_rows: list[tuple] = []
        self._last = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None) -> None:
        self.queries.append((sql, params))
        self._last = self._results.pop(0) if self._results else None

    def executemany(self, sql, rows) -> None:
        self.executemany_rows.extend(rows)

    def fetchone(self):
        return self._last

    def fetchall(self):
        return self._last


class ScriptedConn:
    """One cursor shared by fetch_window; keeps the recorded queries."""

    def __init__(self, results: list | None = None) -> None:
        self._cursor = ScriptedCursor(results or [])
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return self._cursor

    def commit(self) -> None:
        self.committed = True


def _conn() -> ScriptedConn:
    return ScriptedConn(
        [
            (MAX_DT,),  # max sim_ts
            [(1, 0.7, "alert", MAX_DT - 100, 1)],  # predictions
            [],  # features
            [(0.7, "alert", 0.05, 1, 100.0)],  # labelled
        ]
    )


def test_labelled_lookback_reaches_behind_the_label_delay() -> None:
    conn = _conn()
    window = fetch_window(conn, window_sim_days=7, label_delay_days=7)
    labelled_sql, labelled_params = conn._cursor.queries[3]
    assert "JOIN labels" in labelled_sql
    # 7-day window + 7-day delay: look back 14 days, not 7 — the trailing
    # window alone can never hold released labels
    assert labelled_params[0] == MAX_DT - 14 * DAY
    assert labelled_params[1] == MAX_DT  # the released-at filter stays the clock
    assert window["labelled"]


def test_delay_defaults_to_seven_days() -> None:
    conn = _conn()
    fetch_window(conn, window_sim_days=7)
    _, labelled_params = conn._cursor.queries[3]
    assert labelled_params[0] == MAX_DT - 14 * DAY


def test_zero_delay_uses_the_window_alone() -> None:
    conn = _conn()
    fetch_window(conn, window_sim_days=7, label_delay_days=0)
    _, labelled_params = conn._cursor.queries[3]
    assert labelled_params[0] == MAX_DT - 7 * DAY


def test_persist_results_wraps_perf_details_for_jsonb() -> None:
    """psycopg refuses plain dicts at %s placeholders — the details payload
    must travel as Json(...) or every monitor cycle aborts at insert time."""
    from psycopg.types.json import Json

    from fraudops.monitoring.drift import DriftResult, Level
    from fraudops.monitoring.monitor import persist_results
    from fraudops.monitoring.trigger import PerformanceSnapshot

    conn = ScriptedConn()
    score = DriftResult(name="score", psi=0.01, ks_pvalue=None, level=Level.OK)
    perf = PerformanceSnapshot(pr_auc=0.7, cost_per_100k=1_000.0, n_labelled=900)
    alerts = persist_results(conn, 180, [score], score, perf)

    assert alerts == []  # quiet window raises nothing
    perf_row = next(r for r in conn._cursor.executemany_rows if r[1] == "performance")
    assert isinstance(perf_row[6], Json)
    assert perf_row[6].obj == {"pr_auc": 0.7, "cost_per_100k": 1000.0, "n_labelled": 900}
    assert conn.committed
