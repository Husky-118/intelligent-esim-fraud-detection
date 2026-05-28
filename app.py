import streamlit as st
import pandas as pd
import joblib

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="eSIM Fraud Detection",
    layout="wide"
)

# ============================================
# LOAD MODEL
# ============================================

model = joblib.load(
    "output/xgb_fraud_model.pkl"
)

# ============================================
# TITLE
# ============================================

st.title("eSIM Fraud Detection System")

st.write(
    "AI-Based Fraud Detection for 5G Telecom Networks"
)

st.markdown("---")

# ============================================
# USER INPUTS
# ============================================

st.header("Enter Telecom Request Details")

col1, col2 = st.columns(2)

with col1:

    device_id_match = st.selectbox(
        "Device ID Match",
        [0, 1]
    )

    geo_distance_km = st.slider(
        "Geo Distance (KM)",
        0,
        2000,
        50
    )

    request_hour = st.slider(
        "Request Hour",
        0,
        23,
        12
    )

    failed_login_count_24h = st.slider(
        "Failed Logins (24h)",
        0,
        20,
        1
    )

    otp_request_count_1h = st.slider(
        "OTP Requests (1h)",
        0,
        20,
        2
    )

with col2:

    new_device_flag = st.selectbox(
        "New Device Flag",
        [0, 1]
    )

    vpn_proxy_detected = st.selectbox(
        "VPN / Proxy Detected",
        [0, 1]
    )

    impossible_travel_flag = st.selectbox(
        "Impossible Travel Flag",
        [0, 1]
    )

    ip_risk_score = st.slider(
        "IP Risk Score",
        0,
        100,
        20
    )

    behavioral_risk_score = st.slider(
        "Behavioral Risk Score",
        0,
        100,
        25
    )

# ============================================
# DEFAULT VALUES FOR REMAINING FEATURES
# ============================================

input_data = pd.DataFrame([{

    "user_id": 12345,

    "device_id_match": device_id_match,

    "geo_distance_km": geo_distance_km,

    "request_hour": request_hour,

    "days_since_last_request": 10,

    "recent_profile_change": 0,

    "failed_login_count_24h":
        failed_login_count_24h,

    "new_device_flag":
        new_device_flag,

    "otp_request_count_1h":
        otp_request_count_1h,

    "account_age_days": 365,

    "impossible_travel_flag":
        impossible_travel_flag,

    "vpn_proxy_detected":
        vpn_proxy_detected,

    "repeated_request_10min": 0,

    "sim_swap_history_count": 0,

    "ip_risk_score":
        ip_risk_score,

    "behavioral_risk_score":
        behavioral_risk_score,

    "hour": request_hour,

    "day": 15,

    "month": 5,

    "weekday": 2
}])

# ============================================
# PREDICT BUTTON
# ============================================

if st.button("Detect Fraud"):

    fraud_probability = model.predict_proba(
        input_data
    )[0][1]

    prediction = (
        fraud_probability >= 0.10
    )

    st.markdown("---")

    st.subheader("Prediction Result")

    st.write(
        f"Fraud Probability: "
        f"{fraud_probability:.2%}"
    )

    if prediction:

        st.error(
            "High Fraud Risk Detected"
        )

    else:

        st.success(
            "Legitimate Request"
        )