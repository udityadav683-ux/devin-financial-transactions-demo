package com.fraudrisk;

/**
 * Holds a risk score contribution and its associated reason.
 */
public class RiskResult {

    private final double score;
    private final String reason;

    public RiskResult(double score, String reason) {
        this.score = score;
        this.reason = reason;
    }

    public double getScore() {
        return score;
    }

    public String getReason() {
        return reason;
    }
}
