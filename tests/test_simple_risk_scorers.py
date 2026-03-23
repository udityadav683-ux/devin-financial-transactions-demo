"""
Comprehensive tests for compute_amount_risk, compute_type_risk, and assign_risk_level
from fraud_risk_scoring module.
"""

import pytest

from fraud_risk_scoring import assign_risk_level, compute_amount_risk, compute_type_risk


# ---------------------------------------------------------------------------
# compute_amount_risk tests
# ---------------------------------------------------------------------------

class TestComputeAmountRisk:
    """Tests for compute_amount_risk(amount) -> (score, reason)."""

    @pytest.mark.parametrize(
        "amount, expected_score",
        [
            # Above 500,000 threshold
            (500_001, 25.0),
            (1_000_000, 25.0),
            (10_000_000, 25.0),
            # Exact boundary: 500,000 is NOT > 500,000, falls to next tier
            (500_000, 20.0),
            # Above 200,000 threshold
            (200_001, 20.0),
            (300_000, 20.0),
            (499_999, 20.0),
            # Exact boundary: 200,000 falls to next tier
            (200_000, 15.0),
            # Above 100,000 threshold
            (100_001, 15.0),
            (150_000, 15.0),
            (199_999, 15.0),
            # Exact boundary: 100,000 falls to next tier
            (100_000, 10.0),
            # Above 10,000 threshold
            (10_001, 10.0),
            (50_000, 10.0),
            (99_999, 10.0),
            # Exact boundary: 10,000 falls to next tier
            (10_000, 5.0),
            # Above 5,000 threshold
            (5_001, 5.0),
            (7_500, 5.0),
            (9_999, 5.0),
            # Exact boundary: 5,000 falls to else branch
            (5_000, 0.0),
            # Below 5,000 → 0.0
            (4_999, 0.0),
            (1_000, 0.0),
            (1, 0.0),
            (0, 0.0),
        ],
    )
    def test_score_thresholds(self, amount, expected_score):
        score, _reason = compute_amount_risk(amount)
        assert score == expected_score

    @pytest.mark.parametrize(
        "amount, expected_score",
        [
            (-1, 0.0),
            (-100, 0.0),
            (-5_001, 0.0),
            (-1_000_000, 0.0),
        ],
    )
    def test_negative_amounts(self, amount, expected_score):
        score, reason = compute_amount_risk(amount)
        assert score == expected_score
        assert reason == ""

    def test_zero_amount(self):
        score, reason = compute_amount_risk(0)
        assert score == 0.0
        assert reason == ""

    def test_very_large_amount(self):
        score, reason = compute_amount_risk(999_999_999)
        assert score == 25.0
        assert reason != ""

    # Verify reason strings are non-empty for non-zero scores
    @pytest.mark.parametrize(
        "amount",
        [500_001, 200_001, 100_001, 10_001, 5_001],
    )
    def test_reason_nonempty_for_positive_scores(self, amount):
        score, reason = compute_amount_risk(amount)
        assert score > 0.0
        assert reason != ""

    def test_reason_empty_for_zero_score(self):
        _score, reason = compute_amount_risk(100)
        assert reason == ""

    def test_reason_contains_amount_for_high_tier(self):
        """Reason string should contain formatted amount for non-zero scores."""
        score, reason = compute_amount_risk(600_000)
        assert score == 25.0
        assert "600,000.00" in reason

    def test_reason_format_200k_tier(self):
        score, reason = compute_amount_risk(250_000)
        assert score == 20.0
        assert "250,000.00" in reason

    def test_reason_format_100k_tier(self):
        score, reason = compute_amount_risk(150_000)
        assert score == 15.0
        assert "150,000.00" in reason

    def test_reason_format_10k_tier(self):
        score, reason = compute_amount_risk(15_000)
        assert score == 10.0
        assert "15,000.00" in reason

    def test_reason_format_5k_tier(self):
        score, reason = compute_amount_risk(6_000)
        assert score == 5.0
        assert "6,000.00" in reason

    def test_returns_tuple(self):
        result = compute_amount_risk(100)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_score_is_float(self):
        score, _ = compute_amount_risk(100)
        assert isinstance(score, float)

    def test_reason_is_str(self):
        _, reason = compute_amount_risk(100)
        assert isinstance(reason, str)

    @pytest.mark.parametrize(
        "amount",
        [0.01, 5000.01, 10000.01, 100000.01, 200000.01, 500000.01],
    )
    def test_float_boundary_just_above(self, amount):
        """Floating-point amounts just above boundaries should cross the threshold."""
        score, _ = compute_amount_risk(amount)
        expected_map = {
            0.01: 0.0,
            5000.01: 5.0,
            10000.01: 10.0,
            100000.01: 15.0,
            200000.01: 20.0,
            500000.01: 25.0,
        }
        assert score == expected_map[amount]


# ---------------------------------------------------------------------------
# compute_type_risk tests
# ---------------------------------------------------------------------------

class TestComputeTypeRisk:
    """Tests for compute_type_risk(txn_type) -> (score, reason)."""

    @pytest.mark.parametrize(
        "txn_type, expected_score, expected_reason_fragment",
        [
            ("CASH_OUT", 20.0, "CASH_OUT"),
            ("TRANSFER", 15.0, "TRANSFER"),
            ("DEBIT", 5.0, "DEBIT"),
            ("PAYMENT", 0.0, ""),
            ("CASH_IN", 0.0, ""),
        ],
    )
    def test_known_types(self, txn_type, expected_score, expected_reason_fragment):
        score, reason = compute_type_risk(txn_type)
        assert score == expected_score
        if expected_reason_fragment:
            assert expected_reason_fragment in reason
        else:
            assert reason == ""

    @pytest.mark.parametrize(
        "txn_type",
        [
            "UNKNOWN",
            "wire",
            "cash_out",  # lowercase variant
            "Transfer",  # mixed case
            "",
            "REFUND",
            "CHECK",
        ],
    )
    def test_unknown_types_return_zero(self, txn_type):
        score, reason = compute_type_risk(txn_type)
        assert score == 0.0
        assert reason == ""

    def test_returns_tuple(self):
        result = compute_type_risk("CASH_OUT")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_score_is_float(self):
        score, _ = compute_type_risk("TRANSFER")
        assert isinstance(score, float)

    def test_reason_is_str(self):
        _, reason = compute_type_risk("TRANSFER")
        assert isinstance(reason, str)

    def test_cash_out_reason_text(self):
        _, reason = compute_type_risk("CASH_OUT")
        assert reason == "High-risk transaction type: CASH_OUT"

    def test_transfer_reason_text(self):
        _, reason = compute_type_risk("TRANSFER")
        assert reason == "High-risk transaction type: TRANSFER"

    def test_debit_reason_text(self):
        _, reason = compute_type_risk("DEBIT")
        assert reason == "Moderate-risk transaction type: DEBIT"

    def test_payment_empty_reason(self):
        _, reason = compute_type_risk("PAYMENT")
        assert reason == ""

    def test_cash_in_empty_reason(self):
        _, reason = compute_type_risk("CASH_IN")
        assert reason == ""


# ---------------------------------------------------------------------------
# assign_risk_level tests
# ---------------------------------------------------------------------------

class TestAssignRiskLevel:
    """Tests for assign_risk_level(score) -> str."""

    @pytest.mark.parametrize(
        "score, expected_level",
        [
            # HIGH: score > 70
            (70.01, "HIGH"),
            (71, "HIGH"),
            (85, "HIGH"),
            (100, "HIGH"),
            # Exact boundary: 70 is NOT > 70, so MEDIUM
            (70, "MEDIUM"),
            # MEDIUM: 40 <= score <= 70
            (40, "MEDIUM"),
            (40.01, "MEDIUM"),
            (55, "MEDIUM"),
            (69.99, "MEDIUM"),
            # LOW: score < 40
            (39.99, "LOW"),
            (39, "LOW"),
            (20, "LOW"),
            (1, "LOW"),
            (0, "LOW"),
        ],
    )
    def test_risk_level_thresholds(self, score, expected_level):
        assert assign_risk_level(score) == expected_level

    def test_negative_score(self):
        assert assign_risk_level(-10) == "LOW"

    def test_zero_score(self):
        assert assign_risk_level(0) == "LOW"

    def test_very_high_score(self):
        assert assign_risk_level(1000) == "HIGH"

    def test_returns_string(self):
        result = assign_risk_level(50)
        assert isinstance(result, str)

    @pytest.mark.parametrize(
        "score",
        [0, 25, 39.99, 40, 55, 70, 70.01, 100],
    )
    def test_return_value_in_expected_set(self, score):
        assert assign_risk_level(score) in {"LOW", "MEDIUM", "HIGH"}

    def test_boundary_40_is_medium(self):
        """40 is >= 40, so it should be MEDIUM."""
        assert assign_risk_level(40) == "MEDIUM"

    def test_boundary_70_is_medium(self):
        """70 is NOT > 70, so it should be MEDIUM (not HIGH)."""
        assert assign_risk_level(70) == "MEDIUM"

    def test_boundary_just_below_40(self):
        assert assign_risk_level(39.999) == "LOW"

    def test_boundary_just_above_70(self):
        assert assign_risk_level(70.001) == "HIGH"
