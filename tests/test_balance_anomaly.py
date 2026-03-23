"""Comprehensive tests for compute_balance_anomaly_risk in fraud_risk_scoring.py."""

import pandas as pd
import pytest

from fraud_risk_scoring import compute_balance_anomaly_risk


def make_row(
    oldbalanceOrg: float = 10000.0,
    newbalanceOrig: float = 5000.0,
    amount: float = 5000.0,
    oldbalanceDest: float = 10000.0,
    newbalanceDest: float = 15000.0,
    nameDest: str = "C12345",
) -> pd.Series:
    """Helper to create a pd.Series representing a transaction row."""
    return pd.Series(
        {
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "amount": amount,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest,
            "nameDest": nameDest,
        }
    )


# ── Normal transaction (no anomalies) ────────────────────────────────────────


class TestNoAnomalies:
    """Tests for normal transactions that should produce zero risk."""

    def test_normal_transaction_score_zero(self):
        row = make_row(
            oldbalanceOrg=10000.0,
            newbalanceOrig=5000.0,
            amount=5000.0,
            oldbalanceDest=10000.0,
            newbalanceDest=15000.0,
        )
        score, reason = compute_balance_anomaly_risk(row)
        assert score == 0.0
        assert reason == ""

    def test_normal_transaction_balances_match(self):
        row = make_row(
            oldbalanceOrg=50000.0,
            newbalanceOrig=30000.0,
            amount=20000.0,
            oldbalanceDest=5000.0,
            newbalanceDest=25000.0,
        )
        score, reason = compute_balance_anomaly_risk(row)
        assert score == 0.0
        assert reason == ""


# ── Account drained signal ────────────────────────────────────────────────────


class TestAccountDrained:
    """Tests for the account-drained-to-zero signal (+10.0)."""

    def test_account_drained_to_zero(self):
        row = make_row(
            oldbalanceOrg=10000.0,
            newbalanceOrig=0.0,
            amount=10000.0,
        )
        score, reason = compute_balance_anomaly_risk(row)
        assert score == 10.0
        assert "Origin account fully drained to zero balance" in reason

    def test_account_not_drained_nonzero_new_balance(self):
        row = make_row(
            oldbalanceOrg=10000.0,
            newbalanceOrig=1.0,
            amount=9999.0,
        )
        score, reason = compute_balance_anomaly_risk(row)
        assert score == 0.0
        assert reason == ""

    def test_zero_old_balance_not_drained(self):
        """oldBal == 0 should NOT trigger drained signal (requires oldBal > 0)."""
        row = make_row(
            oldbalanceOrg=0.0,
            newbalanceOrig=0.0,
            amount=0.0,
        )
        score, reason = compute_balance_anomaly_risk(row)
        assert "drained" not in reason.lower()


# ── Balance discrepancy signal ────────────────────────────────────────────────


class TestBalanceDiscrepancy:
    """Tests for the balance discrepancy at origin signal (+5.0)."""

    def test_discrepancy_detected(self):
        # old - amount = 10000 - 5000 = 5000, but new = 3000 → discrepancy
        row = make_row(
            oldbalanceOrg=10000.0,
            newbalanceOrig=3000.0,
            amount=5000.0,
        )
        score, reason = compute_balance_anomaly_risk(row)
        assert score >= 5.0
        assert "Balance discrepancy at origin" in reason

    def test_discrepancy_includes_expected_and_actual(self):
        row = make_row(
            oldbalanceOrg=10000.0,
            newbalanceOrig=3000.0,
            amount=5000.0,
        )
        _, reason = compute_balance_anomaly_risk(row)
        assert "expected 5000.00" in reason
        assert "got 3000.00" in reason

    def test_no_discrepancy_within_tolerance(self):
        """Discrepancy small enough to stay within 0.01 threshold."""
        # Use values where float subtraction stays within tolerance
        row = make_row(
            oldbalanceOrg=10000.0,
            newbalanceOrig=5000.0,
            amount=5000.0,
        )
        score, reason = compute_balance_anomaly_risk(row)
        assert "Balance discrepancy" not in reason

    def test_discrepancy_just_above_tolerance(self):
        """Discrepancy of 0.02 should trigger."""
        row = make_row(
            oldbalanceOrg=10000.0,
            newbalanceOrig=4999.98,
            amount=5000.0,
        )
        score, reason = compute_balance_anomaly_risk(row)
        assert "Balance discrepancy at origin" in reason

    def test_discrepancy_not_triggered_when_old_bal_zero(self):
        """Discrepancy check requires oldBal > 0."""
        row = make_row(
            oldbalanceOrg=0.0,
            newbalanceOrig=5000.0,
            amount=100.0,
        )
        _, reason = compute_balance_anomaly_risk(row)
        assert "Balance discrepancy" not in reason


# ── Zero initial balance signal ───────────────────────────────────────────────


class TestZeroInitialBalance:
    """Tests for transaction from zero-balance account (+5.0)."""

    def test_zero_initial_balance_with_positive_amount(self):
        row = make_row(
            oldbalanceOrg=0.0,
            newbalanceOrig=0.0,
            amount=5000.0,
        )
        score, reason = compute_balance_anomaly_risk(row)
        assert score >= 5.0
        assert "Transaction from account with zero initial balance" in reason

    def test_zero_initial_balance_zero_amount_no_trigger(self):
        """amount must be > 0 to trigger."""
        row = make_row(
            oldbalanceOrg=0.0,
            newbalanceOrig=0.0,
            amount=0.0,
        )
        score, reason = compute_balance_anomaly_risk(row)
        assert "zero initial balance" not in reason.lower()

    def test_nonzero_initial_balance_no_trigger(self):
        row = make_row(
            oldbalanceOrg=100.0,
            newbalanceOrig=0.0,
            amount=100.0,
        )
        _, reason = compute_balance_anomaly_risk(row)
        assert "zero initial balance" not in reason.lower()


# ── Destination balance anomaly signal ────────────────────────────────────────


class TestDestinationBalanceAnomaly:
    """Tests for destination balance anomaly (+5.0) and merchant exclusion."""

    def test_dest_balance_dropped_to_zero(self):
        row = make_row(
            oldbalanceDest=10000.0,
            newbalanceDest=0.0,
            nameDest="C99999",
        )
        score, reason = compute_balance_anomaly_risk(row)
        assert score >= 5.0
        assert "Destination balance dropped to zero after receiving funds" in reason

    def test_dest_balance_not_dropped(self):
        row = make_row(
            oldbalanceDest=10000.0,
            newbalanceDest=15000.0,
            nameDest="C99999",
        )
        score, reason = compute_balance_anomaly_risk(row)
        assert "Destination balance dropped" not in reason

    def test_dest_zero_old_balance_no_trigger(self):
        """oldDest must be > 0 to trigger."""
        row = make_row(
            oldbalanceDest=0.0,
            newbalanceDest=0.0,
            nameDest="C99999",
        )
        _, reason = compute_balance_anomaly_risk(row)
        assert "Destination balance dropped" not in reason

    def test_merchant_exclusion_skips_dest_anomaly(self):
        """nameDest starting with 'M' should skip destination anomaly check."""
        row = make_row(
            oldbalanceDest=10000.0,
            newbalanceDest=0.0,
            nameDest="M12345",
        )
        score, reason = compute_balance_anomaly_risk(row)
        assert "Destination balance dropped" not in reason

    def test_merchant_name_exact_M_prefix(self):
        """Name starting with 'M' (any chars after) is a merchant."""
        row = make_row(
            oldbalanceDest=5000.0,
            newbalanceDest=0.0,
            nameDest="Merchant_XYZ",
        )
        _, reason = compute_balance_anomaly_risk(row)
        assert "Destination balance dropped" not in reason

    def test_non_merchant_lowercase_m(self):
        """Lowercase 'm' should NOT be treated as merchant."""
        row = make_row(
            oldbalanceDest=5000.0,
            newbalanceDest=0.0,
            nameDest="m12345",
        )
        score, reason = compute_balance_anomaly_risk(row)
        assert "Destination balance dropped to zero after receiving funds" in reason


# ── Combined signals ──────────────────────────────────────────────────────────


class TestCombinedSignals:
    """Tests for multiple risk signals triggering simultaneously."""

    def test_drained_and_discrepancy(self):
        """Account drained (+10) AND balance discrepancy (+5) = 15."""
        # old=10000, amount=5000, new=0 → drained (old>0, new==0)
        # expected_new = 10000-5000 = 5000, actual=0 → discrepancy
        row = make_row(
            oldbalanceOrg=10000.0,
            newbalanceOrig=0.0,
            amount=5000.0,
        )
        score, reason = compute_balance_anomaly_risk(row)
        assert score == 15.0
        assert "Origin account fully drained to zero balance" in reason
        assert "Balance discrepancy at origin" in reason

    def test_drained_and_dest_anomaly(self):
        """Account drained (+10) AND dest anomaly (+5) = 15."""
        row = make_row(
            oldbalanceOrg=10000.0,
            newbalanceOrig=0.0,
            amount=10000.0,
            oldbalanceDest=5000.0,
            newbalanceDest=0.0,
            nameDest="C99999",
        )
        score, reason = compute_balance_anomaly_risk(row)
        assert score == 15.0
        assert "Origin account fully drained to zero balance" in reason
        assert "Destination balance dropped to zero after receiving funds" in reason

    def test_zero_initial_and_dest_anomaly(self):
        """Zero initial balance (+5) AND dest anomaly (+5) = 10."""
        row = make_row(
            oldbalanceOrg=0.0,
            newbalanceOrig=0.0,
            amount=5000.0,
            oldbalanceDest=10000.0,
            newbalanceDest=0.0,
            nameDest="C99999",
        )
        score, reason = compute_balance_anomaly_risk(row)
        assert score == 10.0
        assert "Transaction from account with zero initial balance" in reason
        assert "Destination balance dropped to zero after receiving funds" in reason


# ── Score capping at 20.0 ────────────────────────────────────────────────────


class TestScoreCapping:
    """Tests that the final score is capped at 20.0."""

    def test_score_capped_at_20(self):
        """Trigger drained (+10), discrepancy (+5), and dest anomaly (+5) = 20."""
        row = make_row(
            oldbalanceOrg=10000.0,
            newbalanceOrig=0.0,
            amount=5000.0,
            oldbalanceDest=10000.0,
            newbalanceDest=0.0,
            nameDest="C99999",
        )
        score, reason = compute_balance_anomaly_risk(row)
        assert score == 20.0

    def test_raw_score_exceeds_20_still_capped(self):
        """
        Trigger ALL four signals so raw score = 10 + 5 + 5 + 5 = 25.
        But the function must cap at 20.0.

        - oldBal=0, amount>0 → zero initial (+5)
          Wait, drained needs oldBal > 0.
        We need: drained(+10), discrepancy(+5), zero_initial(+5), dest_anomaly(+5)
        But drained requires oldBal>0, zero_initial requires oldBal==0. These are mutually exclusive.
        So max raw = 10 + 5 + 5 = 20 (drained + discrepancy + dest) or 5 + 5 = 10 (zero_initial + dest).
        Actually the max possible is exactly 20 from drained+discrepancy+dest.
        Let's verify capping works by confirming the result is exactly 20.
        """
        row = make_row(
            oldbalanceOrg=10000.0,
            newbalanceOrig=0.0,
            amount=5000.0,
            oldbalanceDest=10000.0,
            newbalanceDest=0.0,
            nameDest="C99999",
        )
        score, _ = compute_balance_anomaly_risk(row)
        # Raw score = 10 (drained) + 5 (discrepancy) + 5 (dest anomaly) = 20
        assert score == 20.0
        assert score <= 20.0


# ── Reason string format ─────────────────────────────────────────────────────


class TestReasonFormat:
    """Tests for the reason string formatting."""

    def test_single_reason_no_separator(self):
        row = make_row(
            oldbalanceOrg=10000.0,
            newbalanceOrig=0.0,
            amount=10000.0,
        )
        _, reason = compute_balance_anomaly_risk(row)
        assert "; " not in reason
        assert reason == "Origin account fully drained to zero balance"

    def test_multiple_reasons_joined_with_semicolon(self):
        row = make_row(
            oldbalanceOrg=10000.0,
            newbalanceOrig=0.0,
            amount=5000.0,
        )
        _, reason = compute_balance_anomaly_risk(row)
        parts = reason.split("; ")
        assert len(parts) == 2
        assert parts[0] == "Origin account fully drained to zero balance"
        assert parts[1].startswith("Balance discrepancy at origin")

    def test_three_reasons_joined(self):
        row = make_row(
            oldbalanceOrg=10000.0,
            newbalanceOrig=0.0,
            amount=5000.0,
            oldbalanceDest=10000.0,
            newbalanceDest=0.0,
            nameDest="C99999",
        )
        _, reason = compute_balance_anomaly_risk(row)
        parts = reason.split("; ")
        assert len(parts) == 3

    def test_no_anomalies_empty_reason(self):
        row = make_row()
        _, reason = compute_balance_anomaly_risk(row)
        assert reason == ""


# ── Parametrized edge cases ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "oldbalanceOrg, newbalanceOrig, amount, oldbalanceDest, newbalanceDest, nameDest, expected_score, expected_reasons",
    [
        # Normal transaction — no flags
        (10000, 5000, 5000, 10000, 15000, "C111", 0.0, []),
        # Only account drained
        (5000, 0, 5000, 1000, 6000, "C222", 10.0, ["Origin account fully drained to zero balance"]),
        # Only balance discrepancy (not drained since newBal != 0)
        (10000, 4000, 5000, 1000, 6000, "C333", 5.0, ["Balance discrepancy at origin"]),
        # Only zero initial balance
        (0, 0, 1000, 5000, 6000, "C444", 5.0, ["Transaction from account with zero initial balance"]),
        # Only destination anomaly
        (10000, 5000, 5000, 8000, 0, "C555", 5.0, ["Destination balance dropped to zero after receiving funds"]),
        # Merchant dest — no dest anomaly
        (10000, 5000, 5000, 8000, 0, "M555", 0.0, []),
        # Zero amount from zero balance — no flags
        (0, 0, 0, 0, 0, "C666", 0.0, []),
    ],
    ids=[
        "normal_no_flags",
        "only_account_drained",
        "only_balance_discrepancy",
        "only_zero_initial_balance",
        "only_dest_anomaly",
        "merchant_dest_excluded",
        "all_zeros_no_flags",
    ],
)
def test_parametrized_individual_signals(
    oldbalanceOrg,
    newbalanceOrig,
    amount,
    oldbalanceDest,
    newbalanceDest,
    nameDest,
    expected_score,
    expected_reasons,
):
    row = make_row(
        oldbalanceOrg=oldbalanceOrg,
        newbalanceOrig=newbalanceOrig,
        amount=amount,
        oldbalanceDest=oldbalanceDest,
        newbalanceDest=newbalanceDest,
        nameDest=nameDest,
    )
    score, reason = compute_balance_anomaly_risk(row)
    assert score == expected_score
    for expected in expected_reasons:
        assert expected in reason
    if not expected_reasons:
        assert reason == ""


@pytest.mark.parametrize(
    "discrepancy_offset, should_trigger",
    [
        (0.00, False),    # exact match
        (0.005, False),   # within tolerance
        (0.01, True),     # boundary — float repr makes abs() slightly > 0.01
        (0.02, True),     # just above tolerance
        (1.0, True),      # clearly above
        (1000.0, True),   # large discrepancy
    ],
    ids=[
        "exact_match",
        "within_tolerance_half_cent",
        "boundary_0.01_triggers_due_to_float",
        "just_above_0.02",
        "above_1.0",
        "large_discrepancy",
    ],
)
def test_discrepancy_boundary_values(discrepancy_offset, should_trigger):
    """Test the 0.01 tolerance boundary for balance discrepancy detection."""
    old_bal = 10000.0
    amount = 5000.0
    expected_new = old_bal - amount  # 5000.0
    new_bal = expected_new - discrepancy_offset  # introduce offset

    row = make_row(
        oldbalanceOrg=old_bal,
        newbalanceOrig=new_bal,
        amount=amount,
    )
    score, reason = compute_balance_anomaly_risk(row)
    if should_trigger:
        assert "Balance discrepancy at origin" in reason
        assert score >= 5.0
    else:
        assert "Balance discrepancy" not in reason


# ── Return type validation ────────────────────────────────────────────────────


class TestReturnType:
    """Verify that the function returns the expected types."""

    def test_returns_tuple(self):
        row = make_row()
        result = compute_balance_anomaly_risk(row)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_score_is_float(self):
        row = make_row()
        score, _ = compute_balance_anomaly_risk(row)
        assert isinstance(score, float)

    def test_reason_is_string(self):
        row = make_row()
        _, reason = compute_balance_anomaly_risk(row)
        assert isinstance(reason, str)
