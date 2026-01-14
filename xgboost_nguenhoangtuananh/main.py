# main.py
import warnings
import pandas as pd
import matplotlib.pyplot as plt

# Import từ các module vệ tinh
from data_processing import (
    load_data, perform_downsampling, split_data, 
    calculate_stats, feature_engineering, get_preprocessor
)
from modeling import train_xgboost, find_best_threshold, evaluate_model, plot_feature_importance

# Cấu hình
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
plt.style.use('ggplot')

# ĐƯỜNG DẪN THƯ MỤC CHỨA FILE BASE CỦA BẠN
BASE_DIR = r"C:\Users\ASUS\Downloads\Banks"

def main():
    print("=== PIPELINE XGBOOST FRAUD DETECTION ===")
    
    # 1. Load Data
    try:
        df = load_data(BASE_DIR)
    except Exception as e:
        print(f"Lỗi tải dữ liệu: {e}")
        return

    # 2. Downsampling (1:20)
    df_balanced = perform_downsampling(df)

    # 3. Split Train/Test
    # Lưu ý: Split xong mới tính toán thống kê để tránh Data Leakage
    X_train_raw, X_test_raw, y_train, y_test = split_data(df_balanced)

    # 4. Tính ngưỡng thống kê (Chỉ trên Train)
    stats = calculate_stats(X_train_raw)

    # 5. Feature Engineering (Áp dụng cho cả Train và Test)
    print("\n[BƯỚC 5] TẠO ĐẶC TRƯNG (FEATURE ENGINEERING)...")
    X_train_final = feature_engineering(X_train_raw, stats)
    X_test_final = feature_engineering(X_test_raw, stats)
    print("   -> Hoàn tất tạo đặc trưng.")

    # 6. Chuẩn bị Preprocessor (StandardScaler + OneHot)
    preprocessor = get_preprocessor(X_train_final)

    # 7. Train Model
    pipeline = train_xgboost(X_train_final, y_train, preprocessor)

    # 8. Tìm ngưỡng tối ưu
    y_prob = pipeline.predict_proba(X_test_final)[:, 1]
    best_th = find_best_threshold(y_test, y_prob)

    # 9. Đánh giá chi tiết
    evaluate_model(pipeline, X_test_final, y_test, best_th)

    # 10. Feature Importance
    plot_feature_importance(pipeline)

    print("\n[HOÀN TẤT] Chương trình chạy thành công!")

if __name__ == "__main__":
    main()