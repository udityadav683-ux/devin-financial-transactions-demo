package com.fraudrisk;

/**
 * Represents the final risk assessment for a single transaction.
 */
public class RiskReport {

    private final int transactionId;
    private final double riskScore;
    private final String riskLevel;
    private final String explanation;

    public RiskReport(int transactionId, double riskScore, String riskLevel, String explanation) {
        this.transactionId = transactionId;
        this.riskScore = riskScore;
        this.riskLevel = riskLevel;
        this.explanation = explanation;
    }

    public int getTransactionId() {
        return transactionId;
    }

    public double getRiskScore() {
        return riskScore;
    }

    public String getRiskLevel() {
        return riskLevel;
    }

    public String getExplanation() {
        return explanation;
    }
}
