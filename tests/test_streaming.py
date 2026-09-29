"""Label-delay logic, drift injector and consumer helper tests."""

import numpy as np
import pandas as pd

from fraudops.streaming.labels import available_at_dt, release_due
from fraudops.streaming.replay import Injector, _row_to_event

SEC = 86_400


class TestRowToEvent:
    """Events must carry JSON-native types on BOTH pandas 2 (numpy scalars)
    and pandas 3 (plain Python scalars from iterrows)."""

    def test_python_scalars_stay_numeric(self) -> None:
        row = pd.Series(
            {"TransactionID": 1, "TransactionDT": 86_400, "TransactionAmt": 59.0, "card4": "visa"}
        ).astype(object)
        event = _row_to_event(row)
        assert event["TransactionID"] == 1 and isinstance(event["TransactionID"], int)
        assert event["TransactionDT"] == 86_400 and isinstance(event["TransactionDT"], int)
        assert event["TransactionAmt"] == 59.0 and isinstance(event["TransactionAmt"], float)
        assert event["card4"] == "visa"

    def test_boxed_row_numerics_never_become_strings(self) -> None:
        # a real row reaches _row_to_event via iterrows/iloc boxing: numpy
        # scalars on pandas 2, python floats (ints upcast) on pandas 3 — the
        # contract is "stays JSON-numeric", never str()
        frame = pd.DataFrame(
            {"TransactionID": [1], "TransactionDT": [86_400], "TransactionAmt": [59.0]}
        )
        event = _row_to_event(frame.iloc[0])
        assert event["TransactionDT"] == 86_400
        for key in ("TransactionID", "TransactionDT", "TransactionAmt"):
            assert isinstance(event[key], int | float), f"{key} became {type(event[key])}"

    def test_nan_becomes_null_and_bool_stays_bool(self) -> None:
        row = pd.Series({"card2": float("nan"), "M1": True})
        event = _row_to_event(row)
        assert event["card2"] is None
        assert event["M1"] is True


class TestLabelDelay:
    """The delayed-label contract, against the simulated clock."""

    def test_availability_is_transaction_plus_delay(self) -> None:
        assert available_at_dt(1_000_000, 7) == 1_000_000 + 7 * SEC

    def test_zero_delay_means_immediate(self) -> None:
        assert available_at_dt(500, 0) == 500

    def test_label_not_due_before_clock_passes_availability(self) -> None:
        due_at = available_at_dt(1_000_000, 7)
        assert release_due(due_at, due_at - 1) is False

    def test_label_due_exactly_when_clock_reaches_availability(self) -> None:
        due_at = available_at_dt(1_000_000, 7)
        assert release_due(due_at, due_at) is True

    def test_seven_sim_days_pass_between_transaction_and_release(self) -> None:
        tx_dt = 150 * SEC + 1234
        clock_late = tx_dt + 7 * SEC + 500
        clock_early = tx_dt + 7 * SEC - 500
        assert release_due(available_at_dt(tx_dt, 7), clock_late) is True
        assert release_due(available_at_dt(tx_dt, 7), clock_early) is False


class TestInjector:
    def _injector(self, injections: list[dict]) -> Injector:
        injector = Injector(None, poll_seconds=999)
        injector._injections = injections
        injector._next_poll = float("inf")  # keep the fake list pinned
        return injector

    def test_no_injections_returns_event_unchanged(self) -> None:
        injector = self._injector([])
        event = {"TransactionAmt": 100.0, "card4": "visa"}
        assert injector.apply(event, 170) == event

    def test_amount_factor_applies_from_from_day(self) -> None:
        injector = self._injector(
            [{"kind": "amount_factor", "params": {"factor": 3.0}, "from_day": 165}]
        )
        before = injector.apply({"TransactionAmt": 100.0}, 164)
        after = injector.apply({"TransactionAmt": 100.0}, 165)
        assert before["TransactionAmt"] == 100.0
        assert after["TransactionAmt"] == 300.0

    def test_nullify_removes_only_listed_columns(self) -> None:
        injector = self._injector(
            [{"kind": "nullify", "params": {"columns": ["card4"]}, "from_day": 160}]
        )
        event = injector.apply({"card4": "visa", "C1": 2.0}, 161)
        assert event["card4"] is None
        assert event["C1"] == 2.0

    def test_inactive_day_left_alone(self) -> None:
        injector = self._injector(
            [{"kind": "nullify", "params": {"columns": ["card4"]}, "from_day": 170}]
        )
        assert injector.apply({"card4": "visa"}, 169)["card4"] == "visa"

    def test_does_not_mutate_input_event(self) -> None:
        injector = self._injector(
            [{"kind": "amount_factor", "params": {"factor": 2.0}, "from_day": 0}]
        )
        event = {"TransactionAmt": 50.0}
        shifted = injector.apply(event, 10)
        assert event["TransactionAmt"] == 50.0
        assert shifted["TransactionAmt"] == 100.0


class TestConsumerHelpers:
    def test_top_feature_names_ranked_by_importance(self, trained_bundle) -> None:
        from fraudops.streaming.consumer import top_feature_names

        bundle, _ = trained_bundle
        names = top_feature_names(bundle, 3)
        assert len(names) == 3
        assert set(names) <= set(bundle.pipeline.feature_names_)

    def test_feature_row_values_split_numeric_and_text(self, trained_bundle) -> None:
        from fraudops.streaming.consumer import feature_row_values

        bundle, frame = trained_bundle
        X = bundle.pipeline.transform(frame.head(1))
        rows = list(feature_row_values(X, ["C1", "ProductCD"], 42, 150 * SEC))
        by_name = {r[2]: r for r in rows}
        assert by_name["C1"][3] is not None and by_name["C1"][4] is None
        assert by_name["ProductCD"][3] is None and by_name["ProductCD"][4] in ("W", "H", "C")
        assert all(r[0] == 42 and r[1] == 150 * SEC for r in rows)

    def test_batch_scoring_matches_api_path(self, trained_bundle, registry) -> None:
        """Consumer batches and the serving API score identical rows
        identically (the single-pipeline guarantee, streaming side)."""
        import mlflow

        from fraudops.registry.wrapper import FraudOpsModel
        from fraudops.serving.schemas import Transaction
        from tests.conftest import MODEL_NAME

        _, frame = trained_bundle
        mlflow.set_tracking_uri(registry["uri"])
        bundle = FraudOpsModel.unwrap(mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion"))
        events = [
            Transaction(
                **{
                    k: (
                        int(v)
                        if isinstance(v, np.integer)
                        else float(v)
                        if isinstance(v, np.floating)
                        else v
                    )
                    for k, v in row.items()
                    if k != "isFraud"
                }
            ).model_dump()
            for _, row in frame.head(10).iterrows()
        ]
        X = bundle.pipeline.transform(pd.DataFrame(events))
        scores = bundle.model.booster_.predict(X)
        direct = bundle.score_raw(pd.DataFrame(events))
        np.testing.assert_allclose(scores, direct, rtol=1e-9)
