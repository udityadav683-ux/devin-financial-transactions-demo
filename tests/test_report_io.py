"""
Comprehensive tests for fraud_risk_scoring module.

Covers: load_dataset, generate_risk_report, main
"""

import os
from unittest.mock import patch

import pandas as pd
import pytest

from fraud_risk_scoring import generate_risk_report, load_dataset, main

# ---------------------------------------------------------------------------
# CSV column header used across tests
# ---------------------------------------------------------------------------
CSV_HEADER = (
    "step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,"
    "nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud"
)


def _make_csv(tmp_path, rows, filename="test.csv"):
    """Helper: write a CSV file with the given rows and return its path."""
    path = tmp_path / filename
    path.write_text(CSV_HEADER + "\n" + "\n".join(rows) + "\n")
    return str(path)


def _row(
    step=1,
    txn_type="PAYMENT",
    amount=100.0,
    name_orig="C1000000001",
    old_bal_org=10000.0,
    new_bal_orig=9900.0,
    name_dest="M2000000001",
    old_bal_dest=0.0,
    new_bal_dest=0.0,
    is_fraud=0,
    is_flagged_fraud=0,
):
    """Helper: build a single CSV row string."""
    return (
        f"{step},{txn_type},{amount},{name_orig},{old_bal_org},"
        f"{new_bal_orig},{name_dest},{old_bal_dest},{new_bal_dest},"
        f"{is_fraud},{is_flagged_fraud}"
    )


# ===========================================================================
# load_dataset tests
# ===========================================================================
class TestLoadDataset:
    """Tests for load_dataset()."""

    def test_load_valid_csv(self, tmp_path):
        """Load a valid CSV with known data → verify shape, columns, transaction_id."""
        rows = [
            _row(step=1, txn_type="PAYMENT", amount=100),
            _row(step=2, txn_type="TRANSFER", amount=200),
            _row(step=3, txn_type="CASH_OUT", amount=300),
        ]
        path = _make_csv(tmp_path, rows)
        df = load_dataset(path)

        assert len(df) == 3
        assert "transaction_id" in df.columns
        # Original CSV columns must also be present
        for col in [
            "step", "type", "amount", "nameOrig", "oldbalanceOrg",
            "newbalanceOrig", "nameDest", "oldbalanceDest",
            "newbalanceDest", "isFraud", "isFlaggedFraud",
        ]:
            assert col in df.columns

    def test_empty_csv(self, tmp_path):
        """Empty CSV (header only) → empty DataFrame with correct columns."""
        path = tmp_path / "empty.csv"
        path.write_text(CSV_HEADER + "\n")
        df = load_dataset(str(path))

        assert len(df) == 0
        assert "transaction_id" in df.columns

    def test_transaction_id_sequential(self, tmp_path):
        """Verify transaction_id is sequential starting from 0."""
        rows = [_row() for _ in range(5)]
        path = _make_csv(tmp_path, rows)
        df = load_dataset(path)

        assert list(df["transaction_id"]) == [0, 1, 2, 3, 4]

    def test_single_row(self, tmp_path):
        """Load CSV with a single row."""
        rows = [_row(amount=42.5)]
        path = _make_csv(tmp_path, rows)
        df = load_dataset(path)

        assert len(df) == 1
        assert df.iloc[0]["transaction_id"] == 0
        assert df.iloc[0]["amount"] == 42.5

    def test_preserves_dtypes(self, tmp_path):
        """Numeric columns remain numeric after loading."""
        rows = [_row(amount=1234.56, old_bal_org=5000.0)]
        path = _make_csv(tmp_path, rows)
        df = load_dataset(path)

        assert df["amount"].dtype in ("float64", "int64")
        assert df["oldbalanceOrg"].dtype in ("float64", "int64")


# ===========================================================================
# generate_risk_report tests
# ===========================================================================
class TestGenerateRiskReport:
    """Tests for generate_risk_report()."""

    @staticmethod
    def _make_df(rows_data):
        """Build a DataFrame that mimics load_dataset output."""
        records = []
        for i, r in enumerate(rows_data):
            records.append({"transaction_id": i, **r})
        return pd.DataFrame(records)

    # -- Output shape & columns ------------------------------------------

    def test_output_columns(self):
        """Report must have exactly the four required columns."""
        df = self._make_df([{
            "step": 1, "type": "PAYMENT", "amount": 10,
            "nameOrig": "C1", "oldbalanceOrg": 1000,
            "newbalanceOrig": 990, "nameDest": "M1",
            "oldbalanceDest": 0, "newbalanceDest": 0,
            "isFraud": 0, "isFlaggedFraud": 0,
        }])
        report = generate_risk_report(df)
        assert list(report.columns) == [
            "transaction_id", "risk_score", "risk_level", "explanation",
        ]

    def test_multiple_transactions_count(self):
        """Number of report rows matches input rows."""
        base = {
            "step": 1, "type": "PAYMENT", "amount": 10,
            "nameOrig": "C1", "oldbalanceOrg": 1000,
            "newbalanceOrig": 990, "nameDest": "M1",
            "oldbalanceDest": 0, "newbalanceDest": 0,
            "isFraud": 0, "isFlaggedFraud": 0,
        }
        df = self._make_df([base, base, base, base])
        report = generate_risk_report(df)
        assert len(report) == 4

    # -- Low-risk scenario -----------------------------------------------

    def test_low_risk_small_payment(self):
        """Small PAYMENT → LOW risk, 'No significant risk signals detected'."""
        df = self._make_df([{
            "step": 1, "type": "PAYMENT", "amount": 50,
            "nameOrig": "C100", "oldbalanceOrg": 5000,
            "newbalanceOrig": 4950, "nameDest": "M200",
            "oldbalanceDest": 0, "newbalanceDest": 0,
            "isFraud": 0, "isFlaggedFraud": 0,
        }])
        report = generate_risk_report(df)
        row = report.iloc[0]

        assert row["risk_level"] == "LOW"
        assert row["risk_score"] == 0.0
        assert row["explanation"] == "No significant risk signals detected"

    # -- High-risk scenario ----------------------------------------------

    def test_high_risk_cash_out_drained_fraud(self):
        """Large CASH_OUT + account drained + isFraud → HIGH risk, score > 70."""
        df = self._make_df([{
            "step": 1, "type": "CASH_OUT", "amount": 600000,
            "nameOrig": "C999", "oldbalanceOrg": 600000,
            "newbalanceOrig": 0, "nameDest": "C888",
            "oldbalanceDest": 0, "newbalanceDest": 0,
            "isFraud": 1, "isFlaggedFraud": 0,
        }])
        report = generate_risk_report(df)
        row = report.iloc[0]

        assert row["risk_level"] == "HIGH"
        assert row["risk_score"] > 70

    # -- Score capping ---------------------------------------------------

    def test_score_capped_at_100(self):
        """Combine many risk signals → score must not exceed 100.0."""
        df = self._make_df([{
            "step": 1, "type": "CASH_OUT", "amount": 1000000,
            "nameOrig": "C111", "oldbalanceOrg": 1000000,
            "newbalanceOrig": 0, "nameDest": "C222",
            "oldbalanceDest": 100000, "newbalanceDest": 0,
            "isFraud": 1, "isFlaggedFraud": 1,
        }])
        report = generate_risk_report(df)
        assert report.iloc[0]["risk_score"] <= 100.0

    # -- isFraud and isFlaggedFraud boosts --------------------------------

    def test_is_fraud_boost(self):
        """isFraud == 1 adds +15.0 to score."""
        base = {
            "step": 1, "type": "PAYMENT", "amount": 50,
            "nameOrig": "C100", "oldbalanceOrg": 5000,
            "newbalanceOrig": 4950, "nameDest": "M200",
            "oldbalanceDest": 0, "newbalanceDest": 0,
            "isFlaggedFraud": 0,
        }
        df_no_fraud = self._make_df([{**base, "isFraud": 0}])
        df_fraud = self._make_df([{**base, "isFraud": 1}])

        score_no = generate_risk_report(df_no_fraud).iloc[0]["risk_score"]
        score_yes = generate_risk_report(df_fraud).iloc[0]["risk_score"]

        assert score_yes - score_no == pytest.approx(15.0)

    def test_is_flagged_fraud_boost(self):
        """isFlaggedFraud == 1 adds +10.0 to score."""
        base = {
            "step": 1, "type": "PAYMENT", "amount": 50,
            "nameOrig": "C100", "oldbalanceOrg": 5000,
            "newbalanceOrig": 4950, "nameDest": "M200",
            "oldbalanceDest": 0, "newbalanceDest": 0,
            "isFraud": 0,
        }
        df_no = self._make_df([{**base, "isFlaggedFraud": 0}])
        df_yes = self._make_df([{**base, "isFlaggedFraud": 1}])

        score_no = generate_risk_report(df_no).iloc[0]["risk_score"]
        score_yes = generate_risk_report(df_yes).iloc[0]["risk_score"]

        assert score_yes - score_no == pytest.approx(10.0)

    def test_both_fraud_flags_boost(self):
        """Both isFraud and isFlaggedFraud → combined +25.0 boost."""
        base = {
            "step": 1, "type": "PAYMENT", "amount": 50,
            "nameOrig": "C100", "oldbalanceOrg": 5000,
            "newbalanceOrig": 4950, "nameDest": "M200",
            "oldbalanceDest": 0, "newbalanceDest": 0,
        }
        df_none = self._make_df([{**base, "isFraud": 0, "isFlaggedFraud": 0}])
        df_both = self._make_df([{**base, "isFraud": 1, "isFlaggedFraud": 1}])

        score_none = generate_risk_report(df_none).iloc[0]["risk_score"]
        score_both = generate_risk_report(df_both).iloc[0]["risk_score"]

        assert score_both - score_none == pytest.approx(25.0)

    # -- Explanation content ---------------------------------------------

    def test_explanation_contains_fraud_text(self):
        """When isFraud==1, explanation includes fraud description."""
        df = self._make_df([{
            "step": 1, "type": "PAYMENT", "amount": 50,
            "nameOrig": "C100", "oldbalanceOrg": 5000,
            "newbalanceOrig": 4950, "nameDest": "M200",
            "oldbalanceDest": 0, "newbalanceDest": 0,
            "isFraud": 1, "isFlaggedFraud": 0,
        }])
        explanation = generate_risk_report(df).iloc[0]["explanation"]
        assert "Transaction flagged as fraud in dataset" in explanation

    def test_explanation_contains_flagged_fraud_text(self):
        """When isFlaggedFraud==1, explanation includes business rule text."""
        df = self._make_df([{
            "step": 1, "type": "PAYMENT", "amount": 50,
            "nameOrig": "C100", "oldbalanceOrg": 5000,
            "newbalanceOrig": 4950, "nameDest": "M200",
            "oldbalanceDest": 0, "newbalanceDest": 0,
            "isFraud": 0, "isFlaggedFraud": 1,
        }])
        explanation = generate_risk_report(df).iloc[0]["explanation"]
        assert "Transaction flagged by business fraud detection rules" in explanation

    def test_explanation_high_risk_contains_signals(self):
        """High-risk transaction explanation mentions amount and type signals."""
        df = self._make_df([{
            "step": 1, "type": "CASH_OUT", "amount": 600000,
            "nameOrig": "C999", "oldbalanceOrg": 600000,
            "newbalanceOrig": 0, "nameDest": "C888",
            "oldbalanceDest": 0, "newbalanceDest": 0,
            "isFraud": 0, "isFlaggedFraud": 0,
        }])
        explanation = generate_risk_report(df).iloc[0]["explanation"]
        assert "CASH_OUT" in explanation
        assert "amount" in explanation.lower() or "600" in explanation

    # -- Medium risk level -----------------------------------------------

    def test_medium_risk_level(self):
        """A transaction scoring between 40 and 70 should be MEDIUM."""
        # CASH_OUT (20) + high amount >10k (10) + isFraud (15) = 45 → MEDIUM
        df = self._make_df([{
            "step": 1, "type": "CASH_OUT", "amount": 15000,
            "nameOrig": "C300", "oldbalanceOrg": 20000,
            "newbalanceOrig": 5000, "nameDest": "M400",
            "oldbalanceDest": 0, "newbalanceDest": 0,
            "isFraud": 1, "isFlaggedFraud": 0,
        }])
        report = generate_risk_report(df)
        row = report.iloc[0]
        assert row["risk_level"] == "MEDIUM"
        assert 40 <= row["risk_score"] <= 70


# ===========================================================================
# main() tests
# ===========================================================================
class TestMain:
    """Tests for main()."""

    def test_main_runs_without_error(self, capsys, monkeypatch):
        """main() executes end-to-end without raising exceptions."""
        # Use the actual data/Example1.csv that ships with the repo
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        monkeypatch.chdir(repo_root)

        main()

        captured = capsys.readouterr()
        assert "Loading dataset" in captured.out
        assert "Risk Score Summary" in captured.out

    def test_main_prints_transaction_count(self, capsys, monkeypatch):
        """main() prints the number of loaded transactions."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        monkeypatch.chdir(repo_root)

        main()

        captured = capsys.readouterr()
        assert "Loaded" in captured.out
        assert "transactions" in captured.out

    def test_main_prints_risk_distribution(self, capsys, monkeypatch):
        """main() prints risk level distribution."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        monkeypatch.chdir(repo_root)

        main()

        captured = capsys.readouterr()
        assert "Risk level distribution" in captured.out

    def test_main_saves_report(self, tmp_path, monkeypatch):
        """main() saves the risk report CSV to the expected path."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        monkeypatch.chdir(repo_root)
        output_path = os.path.join(repo_root, "data", "fraud_risk_report.csv")

        main()

        assert os.path.exists(output_path)
        report_df = pd.read_csv(output_path)
        assert "transaction_id" in report_df.columns
        assert "risk_score" in report_df.columns
        assert "risk_level" in report_df.columns

    def test_main_with_mocked_io(self, capsys, tmp_path, monkeypatch):
        """main() works with mocked load_dataset and to_csv."""
        # Create a small test CSV
        csv_content = (
            CSV_HEADER + "\n"
            + _row(step=1, txn_type="PAYMENT", amount=50,
                   name_orig="C1", old_bal_org=1000, new_bal_orig=950,
                   name_dest="M1") + "\n"
        )
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        csv_path = data_dir / "Example1.csv"
        csv_path.write_text(csv_content)

        monkeypatch.chdir(str(tmp_path))

        main()

        captured = capsys.readouterr()
        assert "Loading dataset" in captured.out
        assert "Loaded 1 transactions" in captured.out
        assert "Risk Score Summary" in captured.out
        assert "Total transactions analyzed: 1" in captured.out

        # Verify report file was written
        report_path = data_dir / "fraud_risk_report.csv"
        assert report_path.exists()
        report_df = pd.read_csv(str(report_path))
        assert len(report_df) == 1

    def test_main_prints_report_saved_message(self, capsys, monkeypatch):
        """main() prints the 'report saved' message."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        monkeypatch.chdir(repo_root)

        main()

        captured = capsys.readouterr()
        assert "Risk report saved to" in captured.out
