import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import os

# Import cục bộ
try:
    from feature_engineering import apply_features
    from preprocessing import load_data, preprocess_data
except ImportError:
    # Dự phòng cho thực thi trực tiếp
    from src.feature_engineering import apply_features
    from src.preprocessing import load_data, preprocess_data

def train_model(df=None):
    if df is None:
        df = load_data()
        df = preprocess_data(df)

    # ===============================
    # THÊM CÁC ĐẶC TRƯNG HÀNH VI
    # ===============================
    df = apply_features(df)

    # ===============================
    # CHIA DỮ LIỆU
    # ===============================
    X = df.drop(columns=["fraud_bool"])
    y = df["fraud_bool"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # ===============================
    # ĐẶC TRƯNG PHÂN LOẠI (CÁC CỘT OBJECT)
    # ===============================
    cat_features = np.where(X_train.dtypes == "object")[0]

    # ===============================
    # XỬ LÝ MẤT CÂN BẰNG
    # ===============================
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos

    print(f"Tỷ lệ gian lận: 1 : {scale_pos_weight:.1f}")

    # ===============================
    # HUẤN LUYỆN CATBOOST
    # ===============================
    model = CatBoostClassifier(
        iterations=600,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="Recall",
        scale_pos_weight=scale_pos_weight,
        cat_features=cat_features,
        verbose=100,
        random_seed=42,
        allow_writing_files=False
    )

    print("Đang huấn luyện mô hình...")
    model.fit(X_train, y_train)
    print("Huấn luyện mô hình hoàn tất.")

    return model, X_test, y_test

if __name__ == "__main__":
    train_model()
