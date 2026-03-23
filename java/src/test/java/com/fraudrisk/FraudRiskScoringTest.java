package com.fraudrisk;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class FraudRiskScoringTest {

    private FraudRiskScoring scorer;

    @BeforeEach
    void setUp() {
        scorer = new FraudRiskScoring();
    }

    // --- computeAmountRisk tests ---

    @Test
    void testAmountRiskVeryHigh() {
        RiskResult result = scorer.computeAmountRisk(600000);
        assertEquals(25.0, result.getScore());
        assertTrue(result.getReason().contains("Very high transaction amount"));
    }

    @Test
    void testAmountRiskExtremelyLarge() {
        RiskResult result = scorer.computeAmountRisk(300000);
        assertEquals(20.0, result.getScore());
        assertTrue(result.getReason().contains("Extremely large transaction amount"));
    }

    @Test
    void testAmountRiskVeryLarge() {
        RiskResult result = scorer.computeAmountRisk(150000);
        assertEquals(15.0, result.getScore());
        assertTrue(result.getReason().contains("Very large transaction amount"));
    }

    @Test
    void testAmountRiskHigh() {
        RiskResult result = scorer.computeAmountRisk(50000);
        assertEquals(10.0, result.getScore());
        assertTrue(result.getReason().contains("High transaction amount"));
    }

    @Test
    void testAmountRiskModerate() {
        RiskResult result = scorer.computeAmountRisk(7000);
        assertEquals(5.0, result.getScore());
        assertTrue(result.getReason().contains("Moderate transaction amount"));
    }

    @Test
    void testAmountRiskLow() {
        RiskResult result = scorer.computeAmountRisk(1000);
        assertEquals(0.0, result.getScore());
        assertEquals("", result.getReason());
    }

    // --- computeTypeRisk tests ---

    @Test
    void testTypeRiskCashOut() {
        RiskResult result = scorer.computeTypeRisk("CASH_OUT");
        assertEquals(20.0, result.getScore());
        assertTrue(result.getReason().contains("CASH_OUT"));
    }

    @Test
    void testTypeRiskTransfer() {
        RiskResult result = scorer.computeTypeRisk("TRANSFER");
        assertEquals(15.0, result.getScore());
        assertTrue(result.getReason().contains("TRANSFER"));
    }

    @Test
    void testTypeRiskDebit() {
        RiskResult result = scorer.computeTypeRisk("DEBIT");
        assertEquals(5.0, result.getScore());
        assertTrue(result.getReason().contains("DEBIT"));
    }

    @Test
    void testTypeRiskPayment() {
        RiskResult result = scorer.computeTypeRisk("PAYMENT");
        assertEquals(0.0, result.getScore());
        assertEquals("", result.getReason());
    }

    @Test
    void testTypeRiskCashIn() {
        RiskResult result = scorer.computeTypeRisk("CASH_IN");
        assertEquals(0.0, result.getScore());
        assertEquals("", result.getReason());
    }

    @Test
    void testTypeRiskUnknown() {
        RiskResult result = scorer.computeTypeRisk("OTHER");
        assertEquals(0.0, result.getScore());
        assertEquals("", result.getReason());
    }

    // --- computeBalanceAnomalyRisk tests ---

    @Test
    void testBalanceAnomalyAccountDrained() {
        Transaction txn = createTransaction(0, "TRANSFER", 5000,
                "C1", 5000, 0, "C2", 0, 5000, 0, 0);
        RiskResult result = scorer.computeBalanceAnomalyRisk(txn);
        assertTrue(result.getScore() > 0);
        assertTrue(result.getReason().contains("fully drained"));
    }

    @Test
    void testBalanceAnomalyDiscrepancy() {
        Transaction txn = createTransaction(0, "TRANSFER", 5000,
                "C1", 10000, 3000, "C2", 0, 5000, 0, 0);
        // expected new balance = 10000 - 5000 = 5000, but got 3000
        RiskResult result = scorer.computeBalanceAnomalyRisk(txn);
        assertTrue(result.getScore() >= 5.0);
        assertTrue(result.getReason().contains("Balance discrepancy"));
    }

    @Test
    void testBalanceAnomalyZeroInitialBalance() {
        Transaction txn = createTransaction(0, "PAYMENT", 5000,
                "C1", 0, 0, "M1", 0, 0, 0, 0);
        RiskResult result = scorer.computeBalanceAnomalyRisk(txn);
        assertTrue(result.getScore() >= 5.0);
        assertTrue(result.getReason().contains("zero initial balance"));
    }

    @Test
    void testBalanceAnomalyDestDroppedToZero() {
        Transaction txn = createTransaction(0, "TRANSFER", 5000,
                "C1", 10000, 5000, "C2", 1000, 0, 0, 0);
        RiskResult result = scorer.computeBalanceAnomalyRisk(txn);
        assertTrue(result.getReason().contains("Destination balance dropped to zero"));
    }

    @Test
    void testBalanceAnomalyDestMerchantExcluded() {
        Transaction txn = createTransaction(0, "PAYMENT", 5000,
                "C1", 10000, 5000, "M1234", 1000, 0, 0, 0);
        RiskResult result = scorer.computeBalanceAnomalyRisk(txn);
        assertFalse(result.getReason().contains("Destination balance dropped"));
    }

    @Test
    void testBalanceAnomalyScoreCappedAt20() {
        // Account drained (10) + discrepancy (5) + zero initial (would not apply since oldBal > 0)
        // Let's create a scenario that could exceed 20
        Transaction txn = createTransaction(0, "TRANSFER", 5000,
                "C1", 5000, 0, "C2", 5000, 0, 0, 0);
        // drained = 10, dest dropped = 5, discrepancy possible
        RiskResult result = scorer.computeBalanceAnomalyRisk(txn);
        assertTrue(result.getScore() <= 20.0);
    }

    @Test
    void testBalanceAnomalyNormal() {
        Transaction txn = createTransaction(0, "PAYMENT", 1000,
                "C1", 5000, 4000, "M1", 0, 0, 0, 0);
        RiskResult result = scorer.computeBalanceAnomalyRisk(txn);
        assertEquals(0.0, result.getScore());
        assertEquals("", result.getReason());
    }

    // --- computeRepeatAccountRisk tests ---

    @Test
    void testRepeatAccountRiskWithRepeats() {
        List<Transaction> transactions = new ArrayList<>();
        transactions.add(createTransaction(0, "PAYMENT", 1000,
                "C1", 5000, 4000, "M1", 0, 0, 0, 0));
        transactions.add(createTransaction(1, "PAYMENT", 2000,
                "C1", 4000, 2000, "M2", 0, 0, 0, 0));
        transactions.add(createTransaction(2, "PAYMENT", 500,
                "C2", 3000, 2500, "M3", 0, 0, 0, 0));

        Map<Integer, RiskResult> riskMap = scorer.computeRepeatAccountRisk(transactions);

        // C1 has 2 transactions -> score = min(2*5, 15) = 10
        assertEquals(10.0, riskMap.get(0).getScore());
        assertEquals(10.0, riskMap.get(1).getScore());
        assertTrue(riskMap.get(0).getReason().contains("repeated activity"));

        // C2 has only 1 transaction -> no risk
        assertEquals(0.0, riskMap.get(2).getScore());
    }

    @Test
    void testRepeatAccountRiskCappedAt15() {
        List<Transaction> transactions = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            transactions.add(createTransaction(i, "PAYMENT", 1000,
                    "C1", 5000, 4000, "M1", 0, 0, 0, 0));
        }
        Map<Integer, RiskResult> riskMap = scorer.computeRepeatAccountRisk(transactions);
        // 5 * 5 = 25, but capped at 15
        assertEquals(15.0, riskMap.get(0).getScore());
    }

    // --- computeDestinationRisk tests ---

    @Test
    void testDestinationRiskHighTraffic() {
        List<Transaction> transactions = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            transactions.add(createTransaction(i, "TRANSFER", 1000,
                    "C" + i, 5000, 4000, "C999", 0, 1000, 0, 0));
        }
        Map<Integer, RiskResult> riskMap = scorer.computeDestinationRisk(transactions);
        // C999 receives 4 transactions (> 2) -> score = min(4*3, 10) = 10
        assertTrue(riskMap.get(0).getScore() > 0);
        assertTrue(riskMap.get(0).getReason().contains("high-traffic destination"));
    }

    @Test
    void testDestinationRiskMerchantExcluded() {
        List<Transaction> transactions = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            transactions.add(createTransaction(i, "PAYMENT", 1000,
                    "C" + i, 5000, 4000, "M999", 0, 0, 0, 0));
        }
        Map<Integer, RiskResult> riskMap = scorer.computeDestinationRisk(transactions);
        assertEquals(0.0, riskMap.get(0).getScore());
    }

    @Test
    void testDestinationRiskLowTraffic() {
        List<Transaction> transactions = new ArrayList<>();
        transactions.add(createTransaction(0, "TRANSFER", 1000,
                "C1", 5000, 4000, "C50", 0, 1000, 0, 0));
        transactions.add(createTransaction(1, "TRANSFER", 1000,
                "C2", 5000, 4000, "C51", 0, 1000, 0, 0));
        Map<Integer, RiskResult> riskMap = scorer.computeDestinationRisk(transactions);
        assertEquals(0.0, riskMap.get(0).getScore());
        assertEquals(0.0, riskMap.get(1).getScore());
    }

    // --- computeCashoutPatternRisk tests ---

    @Test
    void testCashoutPatternTransferToKnownCashoutAccount() {
        List<Transaction> transactions = new ArrayList<>();
        // A CASH_OUT by C99 (makes C99 a cashout origin)
        transactions.add(createTransaction(0, "CASH_OUT", 60000,
                "C99", 60000, 0, "C100", 0, 60000, 0, 0));
        // A TRANSFER to C99 (destination is known cashout origin)
        transactions.add(createTransaction(1, "TRANSFER", 50000,
                "C1", 100000, 50000, "C99", 0, 50000, 0, 0));

        Map<Integer, RiskResult> riskMap = scorer.computeCashoutPatternRisk(transactions);
        assertEquals(10.0, riskMap.get(1).getScore());
        assertTrue(riskMap.get(1).getReason().contains("potential layering"));
    }

    @Test
    void testCashoutPatternLargeCashout() {
        List<Transaction> transactions = new ArrayList<>();
        transactions.add(createTransaction(0, "CASH_OUT", 80000,
                "C1", 80000, 0, "C2", 0, 80000, 0, 0));

        Map<Integer, RiskResult> riskMap = scorer.computeCashoutPatternRisk(transactions);
        assertEquals(10.0, riskMap.get(0).getScore());
        assertTrue(riskMap.get(0).getReason().contains("potential fraud cash-out"));
    }

    @Test
    void testCashoutPatternSmallAmountNoRisk() {
        List<Transaction> transactions = new ArrayList<>();
        transactions.add(createTransaction(0, "CASH_OUT", 5000,
                "C1", 5000, 0, "C2", 0, 5000, 0, 0));

        Map<Integer, RiskResult> riskMap = scorer.computeCashoutPatternRisk(transactions);
        assertEquals(0.0, riskMap.get(0).getScore());
    }

    // --- assignRiskLevel tests ---

    @Test
    void testAssignRiskLevelHigh() {
        assertEquals("HIGH", scorer.assignRiskLevel(75));
        assertEquals("HIGH", scorer.assignRiskLevel(100));
    }

    @Test
    void testAssignRiskLevelMedium() {
        assertEquals("MEDIUM", scorer.assignRiskLevel(40));
        assertEquals("MEDIUM", scorer.assignRiskLevel(55));
        assertEquals("MEDIUM", scorer.assignRiskLevel(70));
    }

    @Test
    void testAssignRiskLevelLow() {
        assertEquals("LOW", scorer.assignRiskLevel(0));
        assertEquals("LOW", scorer.assignRiskLevel(39));
    }

    // --- generateRiskReport tests ---

    @Test
    void testGenerateRiskReportBasic() {
        List<Transaction> transactions = new ArrayList<>();
        transactions.add(createTransaction(0, "PAYMENT", 100,
                "C1", 1000, 900, "M1", 0, 0, 0, 0));

        List<RiskReport> reports = scorer.generateRiskReport(transactions);
        assertEquals(1, reports.size());
        assertEquals(0, reports.get(0).getTransactionId());
        assertEquals("LOW", reports.get(0).getRiskLevel());
        assertEquals("No significant risk signals detected", reports.get(0).getExplanation());
    }

    @Test
    void testGenerateRiskReportHighRisk() {
        List<Transaction> transactions = new ArrayList<>();
        // Large CASH_OUT from drained account, flagged as fraud
        transactions.add(createTransaction(0, "CASH_OUT", 600000,
                "C1", 600000, 0, "C2", 0, 600000, 1, 0));

        List<RiskReport> reports = scorer.generateRiskReport(transactions);
        assertEquals(1, reports.size());
        assertEquals("HIGH", reports.get(0).getRiskLevel());
        assertTrue(reports.get(0).getRiskScore() > 70);
    }

    @Test
    void testGenerateRiskReportScoreCappedAt100() {
        List<Transaction> transactions = new ArrayList<>();
        // Max out everything: large amount, CASH_OUT, drained, fraud flagged, business flagged
        transactions.add(createTransaction(0, "CASH_OUT", 600000,
                "C1", 600000, 0, "C2", 5000, 0, 1, 1));

        List<RiskReport> reports = scorer.generateRiskReport(transactions);
        assertTrue(reports.get(0).getRiskScore() <= 100.0);
    }

    @Test
    void testGenerateRiskReportFraudBoost() {
        List<Transaction> transactions = new ArrayList<>();
        transactions.add(createTransaction(0, "PAYMENT", 100,
                "C1", 1000, 900, "M1", 0, 0, 1, 0));

        List<RiskReport> reports = scorer.generateRiskReport(transactions);
        assertTrue(reports.get(0).getRiskScore() >= 15.0);
        assertTrue(reports.get(0).getExplanation().contains("flagged as fraud"));
    }

    @Test
    void testGenerateRiskReportFlaggedFraudBoost() {
        List<Transaction> transactions = new ArrayList<>();
        transactions.add(createTransaction(0, "PAYMENT", 100,
                "C1", 1000, 900, "M1", 0, 0, 0, 1));

        List<RiskReport> reports = scorer.generateRiskReport(transactions);
        assertTrue(reports.get(0).getRiskScore() >= 10.0);
        assertTrue(reports.get(0).getExplanation().contains("business fraud detection"));
    }

    // --- loadDataset and saveReport tests ---

    @Test
    void testLoadDataset(@TempDir Path tempDir) throws Exception {
        Path csvFile = tempDir.resolve("test.csv");
        try (FileWriter fw = new FileWriter(csvFile.toFile())) {
            fw.write("step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud\n");
            fw.write("1,PAYMENT,1000.50,C1,5000,4000,M1,0,0,0,0\n");
            fw.write("1,CASH_OUT,50000,C2,50000,0,C3,0,50000,1,0\n");
        }

        List<Transaction> transactions = scorer.loadDataset(csvFile.toString());
        assertEquals(2, transactions.size());

        assertEquals(0, transactions.get(0).getTransactionId());
        assertEquals("PAYMENT", transactions.get(0).getType());
        assertEquals(1000.50, transactions.get(0).getAmount(), 0.01);
        assertEquals("C1", transactions.get(0).getNameOrig());

        assertEquals(1, transactions.get(1).getTransactionId());
        assertEquals("CASH_OUT", transactions.get(1).getType());
        assertEquals(1, transactions.get(1).getIsFraud());
    }

    @Test
    void testLoadDatasetEmpty(@TempDir Path tempDir) throws Exception {
        Path csvFile = tempDir.resolve("empty.csv");
        try (FileWriter fw = new FileWriter(csvFile.toFile())) {
            fw.write("step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud\n");
        }

        List<Transaction> transactions = scorer.loadDataset(csvFile.toString());
        assertEquals(0, transactions.size());
    }

    @Test
    void testSaveReport(@TempDir Path tempDir) throws Exception {
        Path outputFile = tempDir.resolve("output.csv");
        List<RiskReport> reports = new ArrayList<>();
        reports.add(new RiskReport(0, 45.0, "MEDIUM", "Some risk"));
        reports.add(new RiskReport(1, 10.0, "LOW", "No significant risk signals detected"));

        scorer.saveReport(reports, outputFile.toString());
        assertTrue(outputFile.toFile().exists());
        assertTrue(outputFile.toFile().length() > 0);
    }

    // --- Integration test with full pipeline ---

    @Test
    void testFullPipeline(@TempDir Path tempDir) throws Exception {
        Path csvFile = tempDir.resolve("transactions.csv");
        try (FileWriter fw = new FileWriter(csvFile.toFile())) {
            fw.write("step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud\n");
            fw.write("1,PAYMENT,9839.64,C1231006815,170136,160296.36,M1979787155,0,0,0,0\n");
            fw.write("1,TRANSFER,181,C1305486145,181,0,C553264065,0,0,1,0\n");
            fw.write("1,CASH_OUT,181,C840083671,181,0,C38997010,21182,0,1,0\n");
            fw.write("1,CASH_OUT,229133.94,C905080434,15325,0,C476402209,5083,51513.44,0,0\n");
        }

        List<Transaction> transactions = scorer.loadDataset(csvFile.toString());
        assertEquals(4, transactions.size());

        List<RiskReport> reports = scorer.generateRiskReport(transactions);
        assertEquals(4, reports.size());

        // First transaction is a normal PAYMENT - should be LOW risk
        assertEquals("LOW", reports.get(0).getRiskLevel());

        // Second is a small TRANSFER flagged as fraud - should have elevated risk
        assertTrue(reports.get(1).getRiskScore() > 0);

        // Fourth is a large CASH_OUT - should be HIGH risk
        assertTrue(reports.get(3).getRiskScore() > 40);

        // Save and verify
        Path outputFile = tempDir.resolve("report.csv");
        scorer.saveReport(reports, outputFile.toString());
        assertTrue(outputFile.toFile().exists());
    }

    // --- Helper method ---

    private Transaction createTransaction(int id, String type, double amount,
                                          String nameOrig, double oldBalOrg, double newBalOrig,
                                          String nameDest, double oldBalDest, double newBalDest,
                                          int isFraud, int isFlaggedFraud) {
        Transaction txn = new Transaction();
        txn.setTransactionId(id);
        txn.setStep(1);
        txn.setType(type);
        txn.setAmount(amount);
        txn.setNameOrig(nameOrig);
        txn.setOldbalanceOrg(oldBalOrg);
        txn.setNewbalanceOrig(newBalOrig);
        txn.setNameDest(nameDest);
        txn.setOldbalanceDest(oldBalDest);
        txn.setNewbalanceDest(newBalDest);
        txn.setIsFraud(isFraud);
        txn.setIsFlaggedFraud(isFlaggedFraud);
        return txn;
    }
}
