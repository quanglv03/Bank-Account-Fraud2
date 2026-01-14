import pandas as pd
import numpy as np

def apply_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tinh chỉnh đặc trưng – V2
    Sử dụng số lượng, tỷ lệ và biến đổi log thay vì cờ nhị phân cứng
    """

    df = df.copy()

    # ===============================
    # 1️ BOT / TỰ ĐỘNG HÓA (MỀM)
    # ===============================

    # Độ dài phiên (ngắn hơn = giống bot hơn)
    if "session_length_in_minutes" in df.columns:
        df["session_length_log"] = np.log1p(df["session_length_in_minutes"])

    # Log vận tốc
    if "velocity_6h" in df.columns:
        df["velocity_6h_log"] = np.log1p(df["velocity_6h"])
    if "velocity_24h" in df.columns:
        df["velocity_24h_log"] = np.log1p(df["velocity_24h"])

    # Tỷ lệ vận tốc (hành vi đột biến)
    if "velocity_6h" in df.columns and "velocity_24h" in df.columns:
        df["velocity_ratio_6h_24h"] = df["velocity_6h"] / (df["velocity_24h"] + 1)

    # ===============================
    # 2️ RỦI RO THIẾT BỊ (SỐ LƯỢNG → LOG)
    # ===============================

    if "device_fraud_count" in df.columns:
        df["device_fraud_count_log"] = np.log1p(df["device_fraud_count"])
    if "device_distinct_emails_8w" in df.columns:
        df["device_email_count_log"] = np.log1p(df["device_distinct_emails_8w"])

    # ===============================
    # 3️ DANH TÍNH & EMAIL (TỶ LỆ)
    # ===============================

    if "device_distinct_emails_8w" in df.columns and "bank_months_count" in df.columns:
        df["email_per_device_ratio"] = (
            df["device_distinct_emails_8w"] /
            (df["bank_months_count"] + 1)
        )

    if "name_email_similarity" in df.columns:
        df["email_name_mismatch"] = 1 - df["name_email_similarity"]

    # ===============================
    # 4️ SỰ KHÔNG ỔN ĐỊNH CỦA ĐỊA CHỈ (ĐIỂM SỐ)
    # ===============================

    if "current_address_months_count" in df.columns and "prev_address_months_count" in df.columns:
        df["address_instability_score"] = (
            1 / (df["current_address_months_count"] + 1) +
            1 / (df["prev_address_months_count"] + 1)
        )

    # ===============================
    # 5️ ÁP LỰC TÍN DỤNG (TỶ LỆ)
    # ===============================

    if "proposed_credit_limit" in df.columns and "income" in df.columns:
        df["credit_income_ratio"] = (
            df["proposed_credit_limit"] /
            (df["income"] + 1)
        )

    if "credit_risk_score" in df.columns:
        df["credit_score_inverse"] = 1 / (df["credit_risk_score"] + 1)

    # ===============================
    # 6️ CÁC ĐẶC TRƯNG MỀM KẾT HỢP
    # ===============================

    if "velocity_6h_log" in df.columns and "device_fraud_count_log" in df.columns:
        df["bot_device_risk_score"] = (
            df["velocity_6h_log"] *
            df["device_fraud_count_log"]
        )

    if "email_name_mismatch" in df.columns and "address_instability_score" in df.columns and "credit_score_inverse" in df.columns:
        df["synthetic_identity_score"] = (
            df["email_name_mismatch"] *
            df["address_instability_score"] *
            df["credit_score_inverse"]
        )

    return df
