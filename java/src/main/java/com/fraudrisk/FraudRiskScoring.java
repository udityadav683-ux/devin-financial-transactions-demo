package com.fraudrisk;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;
import com.opencsv.exceptions.CsvValidationException;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringJoiner;

/**
 * Fraud Risk Scoring System for Financial Transactions.
 *
 * Analyzes each transaction independently, identifies risk signals,
 * computes a risk score (0-100), assigns a risk category (LOW, MEDIUM, HIGH),
 * and generates a transaction-level risk report.
 */
public class FraudRiskScoring {

    /**
     * Load the transaction dataset from a CSV file.
     */
    public List<Transaction> loadDataset(String filepath) throws IOException, CsvValidationException {
        List<Transaction> transactions = new ArrayList<>();
        try (CSVReader reader = new CSVReader(new FileReader(filepath))) {
            String[] header = reader.readNext(); // skip header
            if (header == null) {
                return transactions;
            }

            int transactionId = 0;
            String[] line;
            while ((line = reader.readNext()) != null) {
                Transaction txn = new Transaction();
                txn.setTransactionId(transactionId);
                txn.setStep(Integer.parseInt(line[0].trim()));
                txn.setType(line[1].trim());
                txn.setAmount(Double.parseDouble(line[2].trim()));
                txn.setNameOrig(line[3].trim());
                txn.setOldbalanceOrg(Double.parseDouble(line[4].trim()));
                txn.setNewbalanceOrig(Double.parseDouble(line[5].trim()));
                txn.setNameDest(line[6].trim());
                txn.setOldbalanceDest(Double.parseDouble(line[7].trim()));
                txn.setNewbalanceDest(Double.parseDouble(line[8].trim()));
                txn.setIsFraud(Integer.parseInt(line[9].trim()));
                txn.setIsFlaggedFraud(Integer.parseInt(line[10].trim()));
                transactions.add(txn);
                transactionId++;
            }
        }
        return transactions;
    }

    /**
     * Compute risk contribution from transaction amount.
     *
     * Transactions above 10,000 are considered high risk.
     * Score contribution scales with amount magnitude.
     */
    public RiskResult computeAmountRisk(double amount) {
        if (amount > 500000) {
            return new RiskResult(25.0, String.format("Very high transaction amount (>%,.2f)", amount));
        } else if (amount > 200000) {
            return new RiskResult(20.0, String.format("Extremely large transaction amount (%,.2f)", amount));
        } else if (amount > 100000) {
            return new RiskResult(15.0, String.format("Very large transaction amount (%,.2f)", amount));
        } else if (amount > 10000) {
            return new RiskResult(10.0, String.format("High transaction amount (%,.2f)", amount));
        } else if (amount > 5000) {
            return new RiskResult(5.0, String.format("Moderate transaction amount (%,.2f)", amount));
        }
        return new RiskResult(0.0, "");
    }

    /**
     * Compute risk contribution from transaction type.
     *
     * CASH_OUT and TRANSFER are higher risk transaction types.
     */
    public RiskResult computeTypeRisk(String txnType) {
        switch (txnType) {
            case "CASH_OUT":
                return new RiskResult(20.0, "High-risk transaction type: CASH_OUT");
            case "TRANSFER":
                return new RiskResult(15.0, "High-risk transaction type: TRANSFER");
            case "DEBIT":
                return new RiskResult(5.0, "Moderate-risk transaction type: DEBIT");
            case "PAYMENT":
            case "CASH_IN":
                return new RiskResult(0.0, "");
            default:
                return new RiskResult(0.0, "");
        }
    }

    /**
     * Detect unusual balance patterns that may indicate fraud.
     *
     * Checks for:
     * - Account being drained (new balance is 0 after transaction)
     * - Balance mismatch (old balance - amount != new balance)
     * - Destination balance not changing despite receiving funds
     */
    public RiskResult computeBalanceAnomalyRisk(Transaction txn) {
        List<String> reasons = new ArrayList<>();
        double score = 0.0;

        double oldBal = txn.getOldbalanceOrg();
        double newBal = txn.getNewbalanceOrig();
        double amount = txn.getAmount();

        // Account drained to zero
        if (oldBal > 0 && newBal == 0) {
            score += 10.0;
            reasons.add("Origin account fully drained to zero balance");
        }

        // Balance discrepancy at origin
        double expectedNewBal = oldBal - amount;
        if (Math.abs(expectedNewBal - newBal) > 0.01 && oldBal > 0) {
            score += 5.0;
            reasons.add(String.format("Balance discrepancy at origin (expected %.2f, got %.2f)",
                    expectedNewBal, newBal));
        }

        // Origin account has zero balance before transaction
        if (oldBal == 0 && amount > 0) {
            score += 5.0;
            reasons.add("Transaction from account with zero initial balance");
        }

        // Destination balance anomaly: balance doesn't increase after receiving
        double oldDest = txn.getOldbalanceDest();
        double newDest = txn.getNewbalanceDest();
        String nameDest = txn.getNameDest();
        if (!nameDest.startsWith("M")) { // Exclude merchants
            if (oldDest > 0 && newDest == 0) {
                score += 5.0;
                reasons.add("Destination balance dropped to zero after receiving funds");
            }
        }

        return new RiskResult(Math.min(score, 20.0), String.join("; ", reasons));
    }

    /**
     * Identify repeated transactions from the same originating account.
     *
     * Rapid sequences of transactions from the same account increase risk.
     */
    public Map<Integer, RiskResult> computeRepeatAccountRisk(List<Transaction> transactions) {
        // Count transactions per originating account
        Map<String, Integer> accountCounts = new HashMap<>();
        for (Transaction txn : transactions) {
            accountCounts.merge(txn.getNameOrig(), 1, Integer::sum);
        }

        Map<Integer, RiskResult> riskMap = new HashMap<>();
        for (int i = 0; i < transactions.size(); i++) {
            Transaction txn = transactions.get(i);
            int count = accountCounts.getOrDefault(txn.getNameOrig(), 0);
            if (count > 1) {
                double score = Math.min(count * 5.0, 15.0);
                riskMap.put(i, new RiskResult(score,
                        String.format("Account %s has %d transactions in dataset (repeated activity)",
                                txn.getNameOrig(), count)));
            } else {
                riskMap.put(i, new RiskResult(0.0, ""));
            }
        }
        return riskMap;
    }

    /**
     * Assess risk based on destination account patterns.
     *
     * Transactions to frequently targeted destination accounts are riskier.
     */
    public Map<Integer, RiskResult> computeDestinationRisk(List<Transaction> transactions) {
        // Count transactions per destination account
        Map<String, Integer> destCounts = new HashMap<>();
        for (Transaction txn : transactions) {
            destCounts.merge(txn.getNameDest(), 1, Integer::sum);
        }

        Map<Integer, RiskResult> riskMap = new HashMap<>();
        for (int i = 0; i < transactions.size(); i++) {
            Transaction txn = transactions.get(i);
            String nameDest = txn.getNameDest();
            if (nameDest.startsWith("M")) {
                // Merchants are generally less risky
                riskMap.put(i, new RiskResult(0.0, ""));
            } else {
                int count = destCounts.getOrDefault(nameDest, 0);
                if (count > 2) {
                    double score = Math.min(count * 3.0, 10.0);
                    riskMap.put(i, new RiskResult(score,
                            String.format("Destination %s received %d transactions (high-traffic destination)",
                                    nameDest, count)));
                } else {
                    riskMap.put(i, new RiskResult(0.0, ""));
                }
            }
        }
        return riskMap;
    }

    /**
     * Detect high amount followed by cash-out patterns.
     *
     * Fraudulent transactions often involve large transfers followed by
     * immediate cash-out from the destination account.
     */
    public Map<Integer, RiskResult> computeCashoutPatternRisk(List<Transaction> transactions) {
        Map<Integer, RiskResult> riskMap = new HashMap<>();
        for (int i = 0; i < transactions.size(); i++) {
            riskMap.put(i, new RiskResult(0.0, ""));
        }

        // Collect accounts that perform CASH_OUT
        Set<String> cashoutDests = new HashSet<>();
        for (Transaction txn : transactions) {
            if ("CASH_OUT".equals(txn.getType())) {
                cashoutDests.add(txn.getNameOrig());
            }
        }

        // Check transfer rows with amount > 10000
        for (int i = 0; i < transactions.size(); i++) {
            Transaction txn = transactions.get(i);
            String type = txn.getType();
            if (("TRANSFER".equals(type) || "CASH_OUT".equals(type)) && txn.getAmount() > 10000) {
                if ("TRANSFER".equals(type) && cashoutDests.contains(txn.getNameDest())) {
                    riskMap.put(i, new RiskResult(10.0,
                            "Large transfer to account that also performs cash-out (potential layering)"));
                } else if ("CASH_OUT".equals(type) && txn.getAmount() > 50000) {
                    riskMap.put(i, new RiskResult(10.0,
                            String.format("Large cash-out transaction (%,.2f), potential fraud cash-out",
                                    txn.getAmount())));
                }
            }
        }

        return riskMap;
    }

    /**
     * Assign risk category based on score thresholds.
     */
    public String assignRiskLevel(double score) {
        if (score > 70) {
            return "HIGH";
        } else if (score >= 40) {
            return "MEDIUM";
        }
        return "LOW";
    }

    /**
     * Generate a comprehensive transaction-level risk report.
     *
     * Combines all risk signals to compute a final risk score and category
     * for each transaction.
     */
    public List<RiskReport> generateRiskReport(List<Transaction> transactions) {
        // Pre-compute aggregate risk signals
        Map<Integer, RiskResult> repeatRisk = computeRepeatAccountRisk(transactions);
        Map<Integer, RiskResult> destRisk = computeDestinationRisk(transactions);
        Map<Integer, RiskResult> cashoutRisk = computeCashoutPatternRisk(transactions);

        List<RiskReport> results = new ArrayList<>();

        for (int i = 0; i < transactions.size(); i++) {
            Transaction txn = transactions.get(i);
            List<String> explanations = new ArrayList<>();
            double totalScore = 0.0;

            // 1. Amount risk
            RiskResult amtResult = computeAmountRisk(txn.getAmount());
            totalScore += amtResult.getScore();
            if (!amtResult.getReason().isEmpty()) {
                explanations.add(amtResult.getReason());
            }

            // 2. Transaction type risk
            RiskResult typeResult = computeTypeRisk(txn.getType());
            totalScore += typeResult.getScore();
            if (!typeResult.getReason().isEmpty()) {
                explanations.add(typeResult.getReason());
            }

            // 3. Balance anomaly risk
            RiskResult balResult = computeBalanceAnomalyRisk(txn);
            totalScore += balResult.getScore();
            if (!balResult.getReason().isEmpty()) {
                explanations.add(balResult.getReason());
            }

            // 4. Repeated account risk
            RiskResult repResult = repeatRisk.get(i);
            totalScore += repResult.getScore();
            if (!repResult.getReason().isEmpty()) {
                explanations.add(repResult.getReason());
            }

            // 5. Destination risk
            RiskResult dstResult = destRisk.get(i);
            totalScore += dstResult.getScore();
            if (!dstResult.getReason().isEmpty()) {
                explanations.add(dstResult.getReason());
            }

            // 6. Cash-out pattern risk
            RiskResult coResult = cashoutRisk.get(i);
            totalScore += coResult.getScore();
            if (!coResult.getReason().isEmpty()) {
                explanations.add(coResult.getReason());
            }

            // 7. Known fraud indicator boost
            if (txn.getIsFraud() == 1) {
                totalScore += 15.0;
                explanations.add("Transaction flagged as fraud in dataset");
            }
            if (txn.getIsFlaggedFraud() == 1) {
                totalScore += 10.0;
                explanations.add("Transaction flagged by business fraud detection rules");
            }

            // Cap score at 100
            double finalScore = Math.min(Math.round(totalScore * 100.0) / 100.0, 100.0);
            String riskLevel = assignRiskLevel(finalScore);
            String explanation = explanations.isEmpty()
                    ? "No significant risk signals detected"
                    : String.join("; ", explanations);

            results.add(new RiskReport(txn.getTransactionId(), finalScore, riskLevel, explanation));
        }

        return results;
    }

    /**
     * Save the risk report to a CSV file.
     */
    public void saveReport(List<RiskReport> reports, String outputPath) throws IOException {
        try (CSVWriter writer = new CSVWriter(new FileWriter(outputPath))) {
            writer.writeNext(new String[]{"transaction_id", "risk_score", "risk_level", "explanation"});
            for (RiskReport report : reports) {
                writer.writeNext(new String[]{
                        String.valueOf(report.getTransactionId()),
                        String.valueOf(report.getRiskScore()),
                        report.getRiskLevel(),
                        report.getExplanation()
                });
            }
        }
    }

    /**
     * Main entry point for the fraud risk scoring system.
     */
    public static void main(String[] args) {
        String inputPath = "data/Example1.csv";
        String outputPath = "data/fraud_risk_report.csv";

        FraudRiskScoring scorer = new FraudRiskScoring();

        try {
            System.out.printf("Loading dataset from '%s'...%n", inputPath);
            List<Transaction> transactions = scorer.loadDataset(inputPath);
            System.out.printf("Loaded %d transactions.%n", transactions.size());

            System.out.println("\nAnalyzing transactions for fraud risk signals...");
            List<RiskReport> report = scorer.generateRiskReport(transactions);

            // Summary statistics
            System.out.println("\n--- Risk Score Summary ---");
            System.out.printf("Total transactions analyzed: %d%n", report.size());

            // Risk level distribution
            Map<String, Integer> levelCounts = new HashMap<>();
            for (RiskReport r : report) {
                levelCounts.merge(r.getRiskLevel(), 1, Integer::sum);
            }
            System.out.println("Risk level distribution:");
            for (Map.Entry<String, Integer> entry : levelCounts.entrySet()) {
                System.out.printf("  %s: %d%n", entry.getKey(), entry.getValue());
            }

            // Risk score statistics
            double sum = 0, min = Double.MAX_VALUE, max = Double.MIN_VALUE;
            for (RiskReport r : report) {
                sum += r.getRiskScore();
                min = Math.min(min, r.getRiskScore());
                max = Math.max(max, r.getRiskScore());
            }
            double mean = report.isEmpty() ? 0 : sum / report.size();
            System.out.printf("%nRisk score statistics:%n");
            System.out.printf("  Mean: %.2f%n", mean);
            System.out.printf("  Min:  %.2f%n", min);
            System.out.printf("  Max:  %.2f%n", max);

            // Show HIGH risk transactions
            List<RiskReport> highRisk = new ArrayList<>();
            for (RiskReport r : report) {
                if ("HIGH".equals(r.getRiskLevel())) {
                    highRisk.add(r);
                }
            }
            if (!highRisk.isEmpty()) {
                System.out.printf("%n--- HIGH Risk Transactions (%d) ---%n", highRisk.size());
                for (RiskReport r : highRisk) {
                    System.out.printf("  Transaction %d: score=%.2f, explanation=%s%n",
                            r.getTransactionId(), r.getRiskScore(), r.getExplanation());
                }
            }

            // Save report
            scorer.saveReport(report, outputPath);
            System.out.printf("%nRisk report saved to '%s'%n", outputPath);

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
