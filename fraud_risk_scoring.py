"""
Fraud Risk Scoring System for Financial Transactions.

Analyzes each transaction independently, identifies risk signals,
computes a risk score (0-100), assigns a risk category (LOW, MEDIUM, HIGH),
and generates a transaction-level risk report.
"""

import pandas as pd


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load the transaction dataset from a CSV file."""
    df = pd.read_csv(filepath)
    df.index.name = "transaction_id"
    df = df.reset_index()
    return df


def compute_amount_risk(amount: float) -> tuple[float, str]:
    """
    Compute risk contribution from transaction amount.

    Transactions above 10,000 are considered high risk.
    Score contribution scales with amount magnitude.
    """
    if amount > 500000:
        return 25.0, "Very high transaction amount (>{:,.2f})".format(amount)
    elif amount > 200000:
        return 20.0, "Extremely large transaction amount ({:,.2f})".format(amount)
    elif amount > 100000:
        return 15.0, "Very large transaction amount ({:,.2f})".format(amount)
    elif amount > 10000:
        return 10.0, "High transaction amount ({:,.2f})".format(amount)
    elif amount > 5000:
        return 5.0, "Moderate transaction amount ({:,.2f})".format(amount)
    return 0.0, ""


def compute_type_risk(txn_type: str) -> tuple[float, str]:
    """
    Compute risk contribution from transaction type.

    CASH_OUT and TRANSFER are higher risk transaction types.
    """
    risk_map = {
        "CASH_OUT": (20.0, "High-risk transaction type: CASH_OUT"),
        "TRANSFER": (15.0, "High-risk transaction type: TRANSFER"),
        "DEBIT": (5.0, "Moderate-risk transaction type: DEBIT"),
        "PAYMENT": (0.0, ""),
        "CASH_IN": (0.0, ""),
    }
    return risk_map.get(txn_type, (0.0, ""))


def compute_balance_anomaly_risk(row: pd.Series) -> tuple[float, str]:
    """
    Detect unusual balance patterns that may indicate fraud.

    Checks for:
    - Account being drained (new balance is 0 after transaction)
    - Balance mismatch (old balance - amount != new balance)
    - Destination balance not changing despite receiving funds
    """
    reasons = []
    score = 0.0

    old_bal = row["oldbalanceOrg"]
    new_bal = row["newbalanceOrig"]
    amount = row["amount"]

    # Account drained to zero
    if old_bal > 0 and new_bal == 0:
        score += 10.0
        reasons.append("Origin account fully drained to zero balance")

    # Balance discrepancy at origin
    expected_new_bal = old_bal - amount
    if abs(expected_new_bal - new_bal) > 0.01 and old_bal > 0:
        score += 5.0
        reasons.append(
            "Balance discrepancy at origin (expected {:.2f}, got {:.2f})".format(
                expected_new_bal, new_bal
            )
        )

    # Origin account has zero balance before transaction
    if old_bal == 0 and amount > 0:
        score += 5.0
        reasons.append("Transaction from account with zero initial balance")

    # Destination balance anomaly: balance doesn't increase after receiving
    old_dest = row["oldbalanceDest"]
    new_dest = row["newbalanceDest"]
    name_dest = str(row["nameDest"])
    if not name_dest.startswith("M"):  # Exclude merchants
        if old_dest > 0 and new_dest == 0:
            score += 5.0
            reasons.append("Destination balance dropped to zero after receiving funds")

    return min(score, 20.0), "; ".join(reasons)


def compute_repeat_account_risk(
    df: pd.DataFrame,
) -> dict[int, tuple[float, str]]:
    """
    Identify repeated transactions from the same originating account.

    Rapid sequences of transactions from the same account increase risk.
    """
    account_counts = df.groupby("nameOrig").size()
    repeat_accounts = account_counts[account_counts > 1].index.tolist()

    risk_map = {}
    for idx, row in df.iterrows():
        if row["nameOrig"] in repeat_accounts:
            count = account_counts[row["nameOrig"]]
            score = min(count * 5.0, 15.0)
            risk_map[idx] = (
                score,
                "Account {} has {} transactions in dataset (repeated activity)".format(
                    row["nameOrig"], count
                ),
            )
        else:
            risk_map[idx] = (0.0, "")

    return risk_map


def compute_destination_risk(df: pd.DataFrame) -> dict[int, tuple[float, str]]:
    """
    Assess risk based on destination account patterns.

    Transactions to frequently targeted destination accounts are riskier.
    """
    dest_counts = df.groupby("nameDest").size()
    high_traffic_dests = dest_counts[dest_counts > 2].index.tolist()

    risk_map = {}
    for idx, row in df.iterrows():
        name_dest = str(row["nameDest"])
        if name_dest.startswith("M"):
            # Merchants are generally less risky
            risk_map[idx] = (0.0, "")
        elif name_dest in high_traffic_dests:
            count = dest_counts[name_dest]
            score = min(count * 3.0, 10.0)
            risk_map[idx] = (
                score,
                "Destination {} received {} transactions (high-traffic destination)".format(
                    name_dest, count
                ),
            )
        else:
            risk_map[idx] = (0.0, "")

    return risk_map


def compute_cashout_pattern_risk(df: pd.DataFrame) -> dict[int, tuple[float, str]]:
    """
    Detect high amount followed by cash-out patterns.

    Fraudulent transactions often involve large transfers followed by
    immediate cash-out from the destination account.
    """
    risk_map = {idx: (0.0, "") for idx in df.index}

    # Group by step to find transactions in the same time period
    cashout_dests = set(df[df["type"] == "CASH_OUT"]["nameOrig"].tolist())
    transfer_rows = df[
        (df["type"].isin(["TRANSFER", "CASH_OUT"])) & (df["amount"] > 10000)
    ]

    for idx, row in transfer_rows.iterrows():
        if row["type"] == "TRANSFER" and row["nameDest"] in cashout_dests:
            risk_map[idx] = (
                10.0,
                "Large transfer to account that also performs cash-out (potential layering)",
            )
        elif row["type"] == "CASH_OUT" and row["amount"] > 50000:
            risk_map[idx] = (
                10.0,
                "Large cash-out transaction ({:,.2f}), potential fraud cash-out".format(
                    row["amount"]
                ),
            )

    return risk_map


def assign_risk_level(score: float) -> str:
    """Assign risk category based on score thresholds."""
    if score > 70:
        return "HIGH"
    elif score >= 40:
        return "MEDIUM"
    return "LOW"


def generate_risk_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a comprehensive transaction-level risk report.

    Combines all risk signals to compute a final risk score and category
    for each transaction.
    """
    # Pre-compute aggregate risk signals
    repeat_risk = compute_repeat_account_risk(df)
    dest_risk = compute_destination_risk(df)
    cashout_risk = compute_cashout_pattern_risk(df)

    results = []

    for idx, row in df.iterrows():
        explanations = []
        total_score = 0.0

        # 1. Amount risk
        amt_score, amt_reason = compute_amount_risk(row["amount"])
        total_score += amt_score
        if amt_reason:
            explanations.append(amt_reason)

        # 2. Transaction type risk
        type_score, type_reason = compute_type_risk(row["type"])
        total_score += type_score
        if type_reason:
            explanations.append(type_reason)

        # 3. Balance anomaly risk
        bal_score, bal_reason = compute_balance_anomaly_risk(row)
        total_score += bal_score
        if bal_reason:
            explanations.append(bal_reason)

        # 4. Repeated account risk
        rep_score, rep_reason = repeat_risk[idx]
        total_score += rep_score
        if rep_reason:
            explanations.append(rep_reason)

        # 5. Destination risk
        dst_score, dst_reason = dest_risk[idx]
        total_score += dst_score
        if dst_reason:
            explanations.append(dst_reason)

        # 6. Cash-out pattern risk
        co_score, co_reason = cashout_risk[idx]
        total_score += co_score
        if co_reason:
            explanations.append(co_reason)

        # 7. Known fraud indicator boost
        if row.get("isFraud", 0) == 1:
            total_score += 15.0
            explanations.append("Transaction flagged as fraud in dataset")

        if row.get("isFlaggedFraud", 0) == 1:
            total_score += 10.0
            explanations.append("Transaction flagged by business fraud detection rules")

        # Cap score at 100
        final_score = min(round(total_score, 2), 100.0)
        risk_level = assign_risk_level(final_score)
        explanation = (
            "; ".join(explanations)
            if explanations
            else "No significant risk signals detected"
        )

        results.append(
            {
                "transaction_id": row["transaction_id"],
                "risk_score": final_score,
                "risk_level": risk_level,
                "explanation": explanation,
            }
        )

    return pd.DataFrame(results)


def main() -> None:
    """Main entry point for the fraud risk scoring system."""
    input_path = "data/Example1.csv"
    output_path = "data/fraud_risk_report.csv"

    print("Loading dataset from '{}'...".format(input_path))
    df = load_dataset(input_path)
    print("Loaded {} transactions.".format(len(df)))

    print("\nAnalyzing transactions for fraud risk signals...")
    report = generate_risk_report(df)

    # Summary statistics
    print("\n--- Risk Score Summary ---")
    print("Total transactions analyzed: {}".format(len(report)))
    print(
        "Risk level distribution:\n{}".format(
            report["risk_level"].value_counts().to_string()
        )
    )
    print(
        "\nRisk score statistics:\n{}".format(
            report["risk_score"].describe().to_string()
        )
    )

    # Show HIGH risk transactions
    high_risk = report[report["risk_level"] == "HIGH"]
    if not high_risk.empty:
        print("\n--- HIGH Risk Transactions ({}) ---".format(len(high_risk)))
        for _, row in high_risk.iterrows():
            print(
                "  Transaction {}: score={}, explanation={}".format(
                    row["transaction_id"], row["risk_score"], row["explanation"]
                )
            )

    # Save report
    report.to_csv(output_path, index=False)
    print("\nRisk report saved to '{}'".format(output_path))


if __name__ == "__main__":
    main()
