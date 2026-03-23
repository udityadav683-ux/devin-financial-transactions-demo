package com.fraudrisk;

/**
 * Represents a single financial transaction loaded from CSV data.
 */
public class Transaction {

    private int transactionId;
    private int step;
    private String type;
    private double amount;
    private String nameOrig;
    private double oldbalanceOrg;
    private double newbalanceOrig;
    private String nameDest;
    private double oldbalanceDest;
    private double newbalanceDest;
    private int isFraud;
    private int isFlaggedFraud;

    public Transaction() {
    }

    public int getTransactionId() {
        return transactionId;
    }

    public void setTransactionId(int transactionId) {
        this.transactionId = transactionId;
    }

    public int getStep() {
        return step;
    }

    public void setStep(int step) {
        this.step = step;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public double getAmount() {
        return amount;
    }

    public void setAmount(double amount) {
        this.amount = amount;
    }

    public String getNameOrig() {
        return nameOrig;
    }

    public void setNameOrig(String nameOrig) {
        this.nameOrig = nameOrig;
    }

    public double getOldbalanceOrg() {
        return oldbalanceOrg;
    }

    public void setOldbalanceOrg(double oldbalanceOrg) {
        this.oldbalanceOrg = oldbalanceOrg;
    }

    public double getNewbalanceOrig() {
        return newbalanceOrig;
    }

    public void setNewbalanceOrig(double newbalanceOrig) {
        this.newbalanceOrig = newbalanceOrig;
    }

    public String getNameDest() {
        return nameDest;
    }

    public void setNameDest(String nameDest) {
        this.nameDest = nameDest;
    }

    public double getOldbalanceDest() {
        return oldbalanceDest;
    }

    public void setOldbalanceDest(double oldbalanceDest) {
        this.oldbalanceDest = oldbalanceDest;
    }

    public double getNewbalanceDest() {
        return newbalanceDest;
    }

    public void setNewbalanceDest(double newbalanceDest) {
        this.newbalanceDest = newbalanceDest;
    }

    public int getIsFraud() {
        return isFraud;
    }

    public void setIsFraud(int isFraud) {
        this.isFraud = isFraud;
    }

    public int getIsFlaggedFraud() {
        return isFlaggedFraud;
    }

    public void setIsFlaggedFraud(int isFlaggedFraud) {
        this.isFlaggedFraud = isFlaggedFraud;
    }
}
