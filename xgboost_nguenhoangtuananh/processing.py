# data_processing.py
import os
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(base_dir):
    """Tìm và đọc file Base.csv hoặc Excel."""
    print(f"\n[BƯỚC 1] TẢI DỮ LIỆU TỪ: {base_dir}")
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Thư mục không tồn tại: {base_dir}")
        
    files = os.listdir(base_dir)
    base_files = [f for f in files if "base" in f.lower()]

    if len(base_files) == 0:
        raise FileNotFoundError("Không tìm thấy file có tên chứa 'base' trong thư mục.")

    file_name = base_files[0]
    file_path = os.path.join(base_dir, file_name)

    if file_name.lower().endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Định dạng file không hỗ trợ")

    print(f"   -> Đã đọc file: {file_name} | Shape: {df.shape}")
    return df

def perform_downsampling(df):
    """Giảm mẫu Non-Fraud xuống tỷ lệ 1:20 so với Fraud."""
    print("\n[BƯỚC 2] DOWNSAMPLING (Cân bằng dữ liệu)...")
    df_fraud = df[df["fraud_bool"] == 1]
    df_non_fraud = df[df["fraud_bool"] == 0]

    print(f"   -> Fraud gốc: {len(df_fraud)} | Non-Fraud gốc: {len(df_non_fraud)}")

    target_non_fraud = len(df_fraud) * 20
    
    if len(df_non_fraud) > target_non_fraud:
        df_non_fraud_down = resample(
            df_non_fraud,
            replace=False,
            n_samples=target_non_fraud,
            random_state=42
        )
        print(f"   -> Downsample Non-Fraud xuống: {target_non_fraud}")
    else:
        df_non_fraud_down = df_non_fraud
        print("   -> Số lượng Non-Fraud nhỏ hơn mục tiêu, giữ nguyên.")

    df_balanced = pd.concat([df_fraud, df_non_fraud_down])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"   -> Shape sau cân bằng: {df_balanced.shape}")
    return df_balanced

def split_data(df):
    """Chia Train/Test."""
    print("\n[BƯỚC 3] CHIA TRAIN/TEST...")
    X = df.drop(columns=['fraud_bool'])
    y = df['fraud_bool']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"   -> Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def clean_numeric(df_in):
    """Ép kiểu dữ liệu số để tránh lỗi tính toán."""
    num_cols_fix = [
        "prev_address_months_count", "current_address_months_count", "bank_months_count",
        "days_since_request", "session_length_in_minutes",
        "velocity_6h", "velocity_24h", "velocity_4w",
        "device_distinct_emails_8w", "device_fraud_count",
        "name_email_similarity", "proposed_credit_limit", "income"
    ]
    df_out = df_in.copy()
    for col in num_cols_fix:
        if col in df_out.columns:
            df_out[col] = pd.to_numeric(df_out[col], errors="coerce")
    return df_out

def calculate_stats(X_train):
    """Tính các ngưỡng thống kê CHỈ TRÊN TẬP TRAIN."""
    print("\n[BƯỚC 4] TÍNH TOÁN NGƯỠNG THỐNG KÊ (TRAIN ONLY)...")
    X_train = clean_numeric(X_train)
    
    stats = {
        'dsr_q05': X_train["days_since_request"].quantile(0.05),
        'sess_q10': X_train["session_length_in_minutes"].quantile(0.10),
        'vel6_q95': X_train["velocity_6h"].quantile(0.95),
        'vel24_q95': X_train["velocity_24h"].quantile(0.95),
        'vel4w_q95': X_train["velocity_4w"].quantile(0.95)
    }
    print(f"   -> Thresholds: {stats}")
    return stats

def feature_engineering(df_in, thresholds):
    """Tạo đặc trưng mới dựa trên ngưỡng đã tính."""
    df = clean_numeric(df_in)
    
    # Flags thiếu thông tin
    df["no_prev_address"] = (df["prev_address_months_count"] == -1).astype(int)
    df["no_current_address"] = (df["current_address_months_count"] == -1).astype(int)
    df["no_bank_history"] = (df["bank_months_count"] == -1).astype(int)

    # Hành vi & Tốc độ
    df["very_short_time_between_requests"] = (df["days_since_request"] < thresholds['dsr_q05']).astype(int)
    df["very_short_session"] = (df["session_length_in_minutes"] < thresholds['sess_q10']).astype(int)
    df["high_velocity_6h"]  = (df["velocity_6h"]  > thresholds['vel6_q95']).astype(int)
    df["high_velocity_24h"] = (df["velocity_24h"] > thresholds['vel24_q95']).astype(int)
    df["high_velocity_4w"]  = (df["velocity_4w"]  > thresholds['vel4w_q95']).astype(int)

    # Thiết bị & Email
    df["many_emails_same_device"] = (df["device_distinct_emails_8w"] >= 3).astype(int)
    df["device_used_for_fraud_before"] = (df["device_fraud_count"] > 0).astype(int)
    df["low_name_email_similarity"] = (df["name_email_similarity"] < 0.3).astype(int)

    # Rủi ro tài chính
    df["foreign_and_new_account"] = ((df["foreign_request"] == 1) & (df["bank_months_count"] < 3)).astype(int)
    df["high_credit_low_income"] = ((df["proposed_credit_limit"] > 1500) & (df["income"] < 0.3)).astype(int)

    # Risk Score tổng hợp
    risk_features = [
        "no_prev_address", "no_current_address", "no_bank_history",
        "very_short_time_between_requests", "very_short_session",
        "high_velocity_6h", "high_velocity_24h", "high_velocity_4w",
        "many_emails_same_device", "device_used_for_fraud_before",
        "low_name_email_similarity", "foreign_and_new_account",
        "high_credit_low_income"
    ]
    df["risk_score"] = df[risk_features].sum(axis=1)
    
    return df

def get_preprocessor(X_train):
    """Tạo Pipeline tiền xử lý (StandardScaler + OneHot)."""
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X_train.select_dtypes(include=["object"]).columns
    
    print(f"   -> Số lượng cột số: {len(num_cols)}")
    print(f"   -> Số lượng cột category: {len(cat_cols)}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ]
    )
    return preprocessor