"""Labelled-stream visibility for retraining.

``labelled_stream_frame`` is the retrain path's only door to the labels
table, so its visibility rule is pinned here: the live path derives its
cutoff from the release arithmetic itself, and the ``as_of_dt`` replay
(retrospective experiments) must see exactly the labels already visible
at that simulated time — cutoff ``as_of - delay`` — never more.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudops.models import retrain

DAY = 86_400
TRAIN_DAYS, VAL_DAYS, DELAY_DAYS = 10, 2, 7
STREAM_START = (TRAIN_DAYS + VAL_DAYS) * DAY  # day 12


class ScriptedCursor:
    """Returns canned results in execute order."""

    def __init__(self, results: list) -> None:
        self._results = results
        self.executes: list[tuple] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None) -> None:
        self.executes.append((sql, params))
        self._last = self._results.pop(0) if self._results else None

    def fetchone(self):
        return self._last

    def fetchall(self):
        return self._last


class ScriptedConn:
    def __init__(self, results: list) -> None:
        self._cursor = ScriptedCursor(results)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return self._cursor


def _configs(tmp_path):
    cfg = tmp_path / "configs"
    cfg.mkdir(exist_ok=True)
    (cfg / "splits.yaml").write_text(f"train_days: {TRAIN_DAYS}\nval_days: {VAL_DAYS}\n")
    (cfg / "drift.yaml").write_text(f"label_delay_days: {DELAY_DAYS}\n")
    return cfg


@pytest.fixture()
def stream_frame(monkeypatch):
    """Raw stream rows on days 12..16; loader monkeypatched to return them."""
    days = np.arange(12, 17)
    df = pd.DataFrame(
        {
            "TransactionID": np.arange(len(days)) + 1,
            "TransactionDT": days * DAY,
            "TransactionAmt": np.full(len(days), 100.0),
        }
    )
    monkeypatch.setattr(retrain, "load_joined", lambda *a, **k: df)
    return df


def test_as_of_limits_labels_to_the_delay_visibility(tmp_path, stream_frame):
    """A retrain at day 19 sees only day-12 transactions (delay 7)."""
    visible = [(1, 0, 100.0)]  # day 12 only; day 13+ not released yet
    conn = ScriptedConn([visible])
    labelled, cutoff = retrain.labelled_stream_frame(
        conn, tmp_path, _configs(tmp_path), as_of_dt=19 * DAY
    )
    assert cutoff == 19 * DAY - DELAY_DAYS * DAY
    assert labelled["TransactionDT"].max() <= cutoff
    assert list(labelled["TransactionID"]) == [1]
    # the visible-labels query must filter on available_at, not return all
    sql, params = conn._cursor.executes[0]
    assert "available_at_dt <=" in sql
    assert params == (19 * DAY,)


def test_as_of_before_any_label_is_visible_raises(tmp_path, stream_frame):
    """Day 16 + delay 7 -> nothing visible: an explicit error, not silence."""
    conn = ScriptedConn([[]])
    with pytest.raises(RuntimeError, match="no labels visible"):
        retrain.labelled_stream_frame(conn, tmp_path, _configs(tmp_path), as_of_dt=16 * DAY)


def test_live_path_derives_cutoff_from_release_arithmetic(tmp_path, stream_frame):
    """No as_of: cutoff = max(available_at) - delay over ALL released labels."""
    all_labels = [(i + 1, i % 2, 100.0) for i in range(5)]  # through day 16
    # MAX(available_at_dt - delay*DAY): already the latest released dt
    conn = ScriptedConn([(16 * DAY,), all_labels])
    labelled, cutoff = retrain.labelled_stream_frame(conn, tmp_path, _configs(tmp_path))
    assert cutoff == 16 * DAY
    assert list(labelled["TransactionID"]) == [1, 2, 3, 4, 5]
