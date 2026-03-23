"""
Comprehensive tests for aggregate pattern risk functions:
- compute_repeat_account_risk
- compute_destination_risk
- compute_cashout_pattern_risk
"""

import pandas as pd
import pytest

from fraud_risk_scoring import (
    compute_cashout_pattern_risk,
    compute_destination_risk,
    compute_repeat_account_risk,
)

COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
    "transaction_id",
]


def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame from a list of row dicts, filling defaults for missing cols."""
    defaults = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 1000.0,
        "nameOrig": "C1000",
        "oldbalanceOrg": 50000.0,
        "newbalanceOrig": 49000.0,
        "nameDest": "C2000",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 1000.0,
        "isFraud": 0,
        "isFlaggedFraud": 0,
        "transaction_id": 0,
    }
    full_rows = [{**defaults, **r, "transaction_id": i} for i, r in enumerate(rows)]
    return pd.DataFrame(full_rows, columns=COLUMNS)


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """Return an empty DataFrame with the expected columns."""
    return pd.DataFrame(columns=COLUMNS)


# ---------------------------------------------------------------------------
# compute_repeat_account_risk
# ---------------------------------------------------------------------------


class TestComputeRepeatAccountRisk:
    """Tests for compute_repeat_account_risk."""

    def test_empty_dataframe(self, empty_df: pd.DataFrame) -> None:
        result = compute_repeat_account_risk(empty_df)
        assert result == {}

    def test_single_transaction_account_zero_score(self) -> None:
        df = _make_df([{"nameOrig": "C1111"}])
        result = compute_repeat_account_risk(df)
        assert result[0] == (0.0, "")

    def test_two_transactions_same_account(self) -> None:
        df = _make_df([{"nameOrig": "C1111"}, {"nameOrig": "C1111"}])
        result = compute_repeat_account_risk(df)
        expected_score = min(2 * 5.0, 15.0)  # 10.0
        for idx in (0, 1):
            score, reason = result[idx]
            assert score == expected_score
            assert "C1111" in reason
            assert "2 transactions" in reason

    def test_three_transactions_same_account(self) -> None:
        df = _make_df(
            [{"nameOrig": "C2222"}] * 3
        )
        result = compute_repeat_account_risk(df)
        expected_score = min(3 * 5.0, 15.0)  # 15.0
        for idx in range(3):
            assert result[idx][0] == expected_score

    def test_score_capped_at_15(self) -> None:
        """4+ transactions should still cap at 15.0."""
        df = _make_df([{"nameOrig": "C3333"}] * 5)
        result = compute_repeat_account_risk(df)
        for idx in range(5):
            assert result[idx][0] == 15.0

    def test_mix_of_repeated_and_unique_accounts(self) -> None:
        df = _make_df(
            [
                {"nameOrig": "C1000"},
                {"nameOrig": "C1000"},
                {"nameOrig": "C1000"},
                {"nameOrig": "C9999"},
            ]
        )
        result = compute_repeat_account_risk(df)
        # C1000 appears 3 times → 15.0
        assert result[0][0] == 15.0
        assert result[1][0] == 15.0
        assert result[2][0] == 15.0
        # C9999 appears once → 0.0
        assert result[3] == (0.0, "")

    @pytest.mark.parametrize(
        "count, expected_score",
        [
            (1, 0.0),
            (2, 10.0),
            (3, 15.0),
            (4, 15.0),
            (10, 15.0),
        ],
    )
    def test_parametrized_scores(self, count: int, expected_score: float) -> None:
        df = _make_df([{"nameOrig": "CXXX"}] * count)
        result = compute_repeat_account_risk(df)
        assert result[0][0] == expected_score

    def test_reason_format(self) -> None:
        df = _make_df([{"nameOrig": "C5555"}, {"nameOrig": "C5555"}])
        result = compute_repeat_account_risk(df)
        _, reason = result[0]
        assert reason == "Account C5555 has 2 transactions in dataset (repeated activity)"


# ---------------------------------------------------------------------------
# compute_destination_risk
# ---------------------------------------------------------------------------


class TestComputeDestinationRisk:
    """Tests for compute_destination_risk."""

    def test_empty_dataframe(self, empty_df: pd.DataFrame) -> None:
        result = compute_destination_risk(empty_df)
        assert result == {}

    def test_low_traffic_destination_zero_score(self) -> None:
        """Destination with ≤ 2 transactions → 0.0."""
        df = _make_df(
            [{"nameDest": "C8000"}, {"nameDest": "C8000"}]
        )
        result = compute_destination_risk(df)
        for idx in (0, 1):
            assert result[idx] == (0.0, "")

    def test_high_traffic_non_merchant_destination(self) -> None:
        """Destination with count > 2 that is NOT a merchant."""
        df = _make_df([{"nameDest": "C9000"}] * 3)
        result = compute_destination_risk(df)
        expected_score = min(3 * 3.0, 10.0)  # 9.0
        for idx in range(3):
            score, reason = result[idx]
            assert score == expected_score
            assert "C9000" in reason
            assert "3 transactions" in reason

    def test_merchant_destination_always_zero(self) -> None:
        """Merchant destinations (start with 'M') always get 0.0 even if high-traffic."""
        df = _make_df([{"nameDest": "M1000"}] * 5)
        result = compute_destination_risk(df)
        for idx in range(5):
            assert result[idx] == (0.0, "")

    def test_score_capped_at_10(self) -> None:
        """Score should not exceed 10.0 even for very high traffic."""
        df = _make_df([{"nameDest": "C7777"}] * 10)
        result = compute_destination_risk(df)
        for idx in range(10):
            assert result[idx][0] == 10.0

    def test_mix_of_merchant_and_non_merchant(self) -> None:
        df = _make_df(
            [
                {"nameDest": "M5000"},
                {"nameDest": "M5000"},
                {"nameDest": "M5000"},
                {"nameDest": "C6000"},
                {"nameDest": "C6000"},
                {"nameDest": "C6000"},
            ]
        )
        result = compute_destination_risk(df)
        # Merchant: always 0.0
        for idx in range(3):
            assert result[idx] == (0.0, "")
        # Non-merchant with 3 hits: 9.0
        for idx in range(3, 6):
            assert result[idx][0] == 9.0

    @pytest.mark.parametrize(
        "count, expected_score",
        [
            (1, 0.0),
            (2, 0.0),
            (3, 9.0),
            (4, 10.0),
            (5, 10.0),
        ],
    )
    def test_parametrized_scores(self, count: int, expected_score: float) -> None:
        df = _make_df([{"nameDest": "C4444"}] * count)
        result = compute_destination_risk(df)
        assert result[0][0] == expected_score

    def test_reason_format(self) -> None:
        df = _make_df([{"nameDest": "C1234"}] * 4)
        result = compute_destination_risk(df)
        _, reason = result[0]
        assert reason == "Destination C1234 received 4 transactions (high-traffic destination)"


# ---------------------------------------------------------------------------
# compute_cashout_pattern_risk
# ---------------------------------------------------------------------------


class TestComputeCashoutPatternRisk:
    """Tests for compute_cashout_pattern_risk."""

    def test_empty_dataframe(self, empty_df: pd.DataFrame) -> None:
        result = compute_cashout_pattern_risk(empty_df)
        assert result == {}

    def test_transfer_to_cashout_origin_large_amount(self) -> None:
        """TRANSFER with amount > 10000 to a known CASH_OUT origin → 10.0 (layering)."""
        df = _make_df(
            [
                # This CASH_OUT makes "C_CASHER" a cashout origin
                {"type": "CASH_OUT", "amount": 5000.0, "nameOrig": "C_CASHER", "nameDest": "C_X"},
                # TRANSFER to C_CASHER (a cashout origin) with amount > 10000
                {"type": "TRANSFER", "amount": 20000.0, "nameOrig": "C_SENDER", "nameDest": "C_CASHER"},
            ]
        )
        result = compute_cashout_pattern_risk(df)
        score, reason = result[1]
        assert score == 10.0
        assert "potential layering" in reason

    def test_large_cashout_above_50000(self) -> None:
        """CASH_OUT with amount > 50000 → 10.0 (fraud cash-out)."""
        df = _make_df(
            [
                {"type": "CASH_OUT", "amount": 60000.0, "nameOrig": "C_BIG", "nameDest": "C_Y"},
            ]
        )
        result = compute_cashout_pattern_risk(df)
        score, reason = result[0]
        assert score == 10.0
        assert "Large cash-out transaction" in reason
        assert "potential fraud cash-out" in reason

    def test_small_cashout_below_10000(self) -> None:
        """CASH_OUT with amount ≤ 10000 → 0.0 (not in transfer_rows filter)."""
        df = _make_df(
            [
                {"type": "CASH_OUT", "amount": 5000.0, "nameOrig": "C_SMALL", "nameDest": "C_Z"},
            ]
        )
        result = compute_cashout_pattern_risk(df)
        assert result[0] == (0.0, "")

    def test_cashout_between_10000_and_50000(self) -> None:
        """CASH_OUT with 10000 < amount ≤ 50000 → 0.0 (doesn't meet > 50000 threshold)."""
        df = _make_df(
            [
                {"type": "CASH_OUT", "amount": 30000.0, "nameOrig": "C_MID", "nameDest": "C_W"},
            ]
        )
        result = compute_cashout_pattern_risk(df)
        assert result[0] == (0.0, "")

    def test_transfer_dest_not_cashout_origin(self) -> None:
        """TRANSFER with amount > 10000 but dest NOT in cashout origins → 0.0."""
        df = _make_df(
            [
                {"type": "TRANSFER", "amount": 50000.0, "nameOrig": "C_A", "nameDest": "C_NOBODY"},
            ]
        )
        result = compute_cashout_pattern_risk(df)
        assert result[0] == (0.0, "")

    def test_payment_always_zero(self) -> None:
        """PAYMENT transactions always get 0.0."""
        df = _make_df(
            [
                {"type": "PAYMENT", "amount": 100000.0, "nameOrig": "C_PAY", "nameDest": "M_SHOP"},
            ]
        )
        result = compute_cashout_pattern_risk(df)
        assert result[0] == (0.0, "")

    def test_transfer_below_10000_to_cashout_origin(self) -> None:
        """TRANSFER with amount ≤ 10000 to cashout origin → 0.0 (below amount filter)."""
        df = _make_df(
            [
                {"type": "CASH_OUT", "amount": 5000.0, "nameOrig": "C_CASHER2", "nameDest": "C_Q"},
                {"type": "TRANSFER", "amount": 9000.0, "nameOrig": "C_LOW", "nameDest": "C_CASHER2"},
            ]
        )
        result = compute_cashout_pattern_risk(df)
        assert result[1] == (0.0, "")

    def test_reason_format_layering(self) -> None:
        df = _make_df(
            [
                {"type": "CASH_OUT", "amount": 1000.0, "nameOrig": "C_CO", "nameDest": "C_D"},
                {"type": "TRANSFER", "amount": 15000.0, "nameOrig": "C_TR", "nameDest": "C_CO"},
            ]
        )
        result = compute_cashout_pattern_risk(df)
        assert result[1] == (
            10.0,
            "Large transfer to account that also performs cash-out (potential layering)",
        )

    def test_reason_format_large_cashout(self) -> None:
        df = _make_df(
            [
                {"type": "CASH_OUT", "amount": 75000.0, "nameOrig": "C_LG", "nameDest": "C_D2"},
            ]
        )
        result = compute_cashout_pattern_risk(df)
        score, reason = result[0]
        assert score == 10.0
        assert "75,000.00" in reason

    @pytest.mark.parametrize(
        "txn_type, amount, setup_cashout_origin, expected_score",
        [
            ("TRANSFER", 20000.0, True, 10.0),
            ("TRANSFER", 20000.0, False, 0.0),
            ("CASH_OUT", 60000.0, False, 10.0),
            ("CASH_OUT", 30000.0, False, 0.0),
            ("CASH_OUT", 5000.0, False, 0.0),
            ("PAYMENT", 100000.0, False, 0.0),
        ],
    )
    def test_parametrized_scenarios(
        self,
        txn_type: str,
        amount: float,
        setup_cashout_origin: bool,
        expected_score: float,
    ) -> None:
        rows: list[dict] = []
        dest = "C_TARGET"
        if setup_cashout_origin:
            # Add a CASH_OUT from C_TARGET so it becomes a cashout origin
            rows.append(
                {"type": "CASH_OUT", "amount": 1000.0, "nameOrig": dest, "nameDest": "C_SINK"}
            )
        rows.append(
            {"type": txn_type, "amount": amount, "nameOrig": "C_MAIN", "nameDest": dest}
        )
        df = _make_df(rows)
        target_idx = len(rows) - 1
        result = compute_cashout_pattern_risk(df)
        assert result[target_idx][0] == expected_score
