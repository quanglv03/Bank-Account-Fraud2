"""
PIPELINE CHÍNH – DỰ ÁN PHÁT HIỆN GIAN LẬN
Chạy toàn bộ phân tích + mô hình hóa + đánh giá
"""

import sys
import os

# Thêm src vào python path nếu chưa có
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import load_data, preprocess_data
import eda
import model_training
import Bank_2.src.catboost_levanquang.evaluation as evaluation

print("\n==============================")
print(" PHÁT HIỆN GIAN LẬN – TOÀN BỘ PIPELINE")
print("==============================\n")

# ===============================
# 1. TẢI DỮ LIỆU
# ===============================
print("Bước 1: Đang tải dữ liệu...")
df = load_data()
df = preprocess_data(df)
print("✔ Dữ liệu đã được tải thành công\n")

# ===============================
# 2. EDA
# ===============================
print("Bước 2: Đang chạy Phân tích Dữ liệu Khám phá (EDA)...")
eda.run_all_eda(df)
print("✔ EDA hoàn tất\n")

# ===============================
# 3. HUẤN LUYỆN MÔ HÌNH
# ===============================
print("Bước 3: Đang huấn luyện mô hình gian lận CatBoost...")
model, X_test, y_test = model_training.train_model(df)
print("✔ Huấn luyện mô hình hoàn tất\n")

# ===============================
# 4. ĐÁNH GIÁ
# ===============================
print("Bước 4: Đang đánh giá mô hình...")
evaluation.evaluate_model(model, X_test, y_test)
evaluation.search_threshold(model, X_test, y_test)

print("\n==============================")
print(" TẤT CẢ CÁC BƯỚC ĐÃ HOÀN THÀNH THÀNH CÔNG")
print("==============================")
