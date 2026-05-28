import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# -----------------------------
# RANDOM SEED
# -----------------------------
np.random.seed(42)

# -----------------------------
# CONFIGURATION
# -----------------------------
N_ROWS = 10000

START_DATE = datetime(2025, 1, 1)

data = []

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def random_timestamp():

    return START_DATE + timedelta(
        minutes=np.random.randint(0, 525600)
    )


def weighted_request_hour():

    probabilities = [
        0.01, 0.01, 0.01, 0.01, 0.01, 0.02,
        0.04, 0.05, 0.06, 0.07, 0.08, 0.08,
        0.08, 0.08, 0.07, 0.07, 0.06, 0.05,
        0.05, 0.04, 0.03, 0.02, 0.01, 0.01
    ]

    probabilities = np.array(probabilities)

    probabilities = probabilities / probabilities.sum()

    hour = np.random.choice(
        range(24),
        p=probabilities
    )

    return hour


# -----------------------------
# DATA GENERATION
# -----------------------------

for _ in range(N_ROWS):

    user_id = np.random.randint(10000, 99999)

    timestamp = random_timestamp()

    request_hour = weighted_request_hour()

    # -----------------------------
    # BASE USER BEHAVIOR
    # -----------------------------

    device_id_match = np.random.binomial(1, 0.78)

    geo_distance_km = abs(
        np.random.normal(35, 30)
    )

    days_since_last_request = max(
        0,
        np.random.poisson(18)
    )

    recent_profile_change = np.random.binomial(1, 0.12)

    failed_login_count_24h = np.random.poisson(1.3)

    new_device_flag = np.random.binomial(1, 0.18)

    otp_request_count_1h = np.random.poisson(2)

    account_age_days = np.random.randint(30, 2500)

    impossible_travel_flag = np.random.binomial(1, 0.02)

    vpn_proxy_detected = np.random.binomial(1, 0.08)

    repeated_request_10min = np.random.binomial(1, 0.05)

    sim_swap_history_count = np.random.poisson(0.3)

    ip_risk_score = np.random.randint(1, 45)

    # -----------------------------
    # ADD RANDOM REALISTIC BEHAVIOR
    # -----------------------------

    # Some legit users travel
    if np.random.rand() < 0.08:
        geo_distance_km += np.random.randint(100, 1200)

    # Password mistakes
    if np.random.rand() < 0.10:
        failed_login_count_24h += np.random.randint(2, 6)

    # Heavy OTP usage
    if np.random.rand() < 0.05:
        otp_request_count_1h += np.random.randint(3, 8)

    # Some users active at night
    if np.random.rand() < 0.07:
        request_hour = np.random.choice(
            [0, 1, 2, 3, 4, 23]
        )

    # -----------------------------
    # FRAUD RISK ENGINE
    # -----------------------------

    risk_score = 0

    # Device mismatch
    if device_id_match == 0:
        risk_score += 15

    # Impossible travel
    if impossible_travel_flag:
        risk_score += 25

    # VPN / proxy usage
    if vpn_proxy_detected:
        risk_score += 15

    # New device
    if new_device_flag:
        risk_score += 10

    # Failed login spikes
    if failed_login_count_24h >= 4:
        risk_score += 15

    # OTP abuse
    if otp_request_count_1h >= 5:
        risk_score += 20

    # Very high geo distance
    if geo_distance_km > 500:
        risk_score += 15

    # Recent profile changes
    if recent_profile_change:
        risk_score += 10

    # Repeated provisioning attempts
    if repeated_request_10min:
        risk_score += 20

    # Suspicious IP score
    if ip_risk_score > 35:
        risk_score += 10

    # SIM swap history
    if sim_swap_history_count >= 2:
        risk_score += 10

    # Very new account
    if account_age_days < 30:
        risk_score += 10

    # -----------------------------
    # FRAUD LABEL LOGIC
    # -----------------------------

    fraud_probability = min(
        0.95,
        risk_score / 100
    )

    label = np.random.binomial(
        1,
        fraud_probability * 0.35
    )

    # -----------------------------
    # STORE RECORD
    # -----------------------------

    data.append([
        user_id,
        timestamp,
        device_id_match,
        round(geo_distance_km, 2),
        request_hour,
        days_since_last_request,
        recent_profile_change,
        failed_login_count_24h,
        new_device_flag,
        otp_request_count_1h,
        account_age_days,
        impossible_travel_flag,
        vpn_proxy_detected,
        repeated_request_10min,
        sim_swap_history_count,
        ip_risk_score,
        risk_score,
        label
    ])

# -----------------------------
# CREATE DATAFRAME
# -----------------------------

columns = [
    "user_id",
    "timestamp",
    "device_id_match",
    "geo_distance_km",
    "request_hour",
    "days_since_last_request",
    "recent_profile_change",
    "failed_login_count_24h",
    "new_device_flag",
    "otp_request_count_1h",
    "account_age_days",
    "impossible_travel_flag",
    "vpn_proxy_detected",
    "repeated_request_10min",
    "sim_swap_history_count",
    "ip_risk_score",
    "behavioral_risk_score",
    "label"
]

df = pd.DataFrame(data, columns=columns)

# -----------------------------
# SHUFFLE DATASET
# -----------------------------

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# -----------------------------
# SAVE CSV
# -----------------------------

df.to_csv(
    "synthetic_esim_fraud.csv",
    index=False
)

# -----------------------------
# OUTPUT
# -----------------------------

print("\nDataset Shape:")
print(df.shape)

print("\nFraud Distribution:")
print(df["label"].value_counts(normalize=True))

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSample Rows:")
print(df.head())