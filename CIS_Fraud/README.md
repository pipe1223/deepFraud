# 📌 IEEE-CIS Fraud Detection Dataset

The **IEEE-CIS Fraud Detection Dataset** was released as part of a Kaggle competition and is designed for detecting fraudulent transactions. It contains **two primary files**:

- **`train_transaction.csv`** – Transactional data (e.g., purchase amount, card type, device info).
- **`train_identity.csv`** – Additional identity-related information (e.g., IP address, browser, device type).

The dataset also includes a **test set** (`test_transaction.csv` and `test_identity.csv`) where fraud labels are missing.

---

## 📂 Dataset Structure

### 1️⃣ Transactional Data (`train_transaction.csv`)

Each row represents a transaction, and the dataset includes **394,000+ transactions** with the following columns:

| Column Name       | Description |
|-------------------|-------------|
| `TransactionID`   | Unique identifier for each transaction |
| `isFraud`         | **Target variable** (1 = Fraud, 0 = Legitimate) |
| `TransactionDT`   | Transaction time (relative, not actual timestamp) |
| `TransactionAmt`  | Transaction amount ($) |
| `ProductCD`       | Product category for the purchase |
| `card1`-`card6`   | Credit card details (hashed) |
| `addr1`, `addr2`  | Billing address (hashed) |
| `dist1`, `dist2`  | Distance metrics between transactions |
| `P_emaildomain`   | Purchaser’s email domain |
| `R_emaildomain`   | Recipient’s email domain |
| `C1`-`C14`        | Count-based features (e.g., # of transactions by user) |
| `D1`-`D15`        | Time-based features (e.g., days since last transaction) |
| `M1`-`M9`         | Boolean values indicating missing fields |

---

### 2️⃣ Identity Data (`train_identity.csv`)

This file contains additional **device and network** information for about **30% of transactions** in `train_transaction.csv`.

| Column Name      | Description |
|-----------------|-------------|
| `TransactionID` | Unique transaction identifier (to merge with `train_transaction.csv`) |
| `DeviceType`    | Type of device (desktop, mobile, etc.) |
| `DeviceInfo`    | Browser/device details |
| `id_01`-`id_38` | Various identity attributes (hashed) |
| `IP address`    | Partial hashed IP addresses (`id_30` = OS, `id_31` = browser) |

---

## 📊 Key Characteristics

✔ **Highly Imbalanced:** Only **3.5% of transactions** are fraudulent (major class imbalance).  
✔ **Feature Engineering Needed:** Many categorical & anonymized features require transformations.  
✔ **Time-Series Nature:** `TransactionDT` allows for time-based modeling.  
✔ **Missing Data:** Many `train_identity.csv` values are missing (only ~30% of transactions have identity data).

---
