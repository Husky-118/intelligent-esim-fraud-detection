## Intelligent eSIM Provisioning Fraud Detection Using AI/ML for 5G Telecom Networks

# Overview

This project is an AI-driven telecom fraud detection prototype designed to identify fraudulent eSIM provisioning requests in 5G telecom environments.
The system combines:
* Machine Learning
* Behavioral Analytics
* Threshold Tuning
* Telecom Risk Indicators
to detect modern eSIM-based fraud attempts such as digital SIM swap attacks and OTP interception fraud.

---

# Problem Statement
eSIM provisioning enables users to activate SIM profiles digitally without physical SIM cards.
However, this introduces new cybersecurity risks where attackers can hijack phone numbers remotely and intercept OTPs for banking or UPI fraud.
Traditional rule-based systems struggle to adapt to evolving attack patterns. 
This project demonstrates how AI/ML can improve adaptive telecom fraud detection.

---
# Real-World Impact

This project demonstrates how AI-driven behavioral analytics can improve fraud detection in modern telecom environments
where traditional rule-based systems often fail to detect evolving attack patterns. By combining machine learning,
threshold optimization, and telecom-specific risk indicators, the prototype simulates how adaptive fraud detection systems
can help reduce fraudulent eSIM provisioning attempts, OTP interception attacks, and digital SIM swap fraud in 5G networks. 
The project highlights the importance of minimizing false negatives in cybersecurity systems where undetected fraud can lead to 
severe financial and operational impact.

---

# Technologies Used
* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SMOTE
* Streamlit
* Matplotlib

---

# Dataset Features
The synthetic telecom fraud dataset includes features such as:
* device_id_match
* geo_distance_km
* request_hour
* failed_login_count_24h
* otp_request_count_1h
* impossible_travel_flag
* vpn_proxy_detected
* ip_risk_score
* behavioral_risk_score

---

# Machine Learning Models
Two models were trained and compared:
* Random Forest
* XGBoost
XGBoost performed better for fraud detection due to higher recall and lower false negative rates.

---

# Threshold Tuning
Threshold tuning was applied to optimize fraud recall and reduce false negatives.
This improved:
* Fraud sensitivity
* Recall
* Detection capability for high-risk telecom fraud scenarios

---

# Streamlit Dashboard
The project includes an interactive Streamlit dashboard for:
* Fraud probability prediction
* Telecom risk analysis
* Behavioral fraud assessment

---

# Project Structure
```bash
esim-fraud-detection/
│
├── app.py
├── generate_dataset.py
├── threshold_tuning.py
├── synthetic_esim_fraud.csv
├── requirements.txt
│
├── output/
│   ├── xgb_fraud_model.pkl
│   ├── threshold_tuning_results.csv
│   ├── feature_importance.csv
│
├── screenshots/
│
└── report/

---

# Cybersecurity Concepts Demonstrated
* Behavioral Analytics
* Fraud Detection
* Telecom Security
* Threshold Optimization
* Risk Scoring
* Anomaly Detection
* Imbalanced Learning

---

# Future Scope
* Real telecom data integration
* Deep learning models
* SIEM integration
* Cloud deployment
* Real-time telecom monitoring

---
Author
Mounika G
a G
Mounika G
