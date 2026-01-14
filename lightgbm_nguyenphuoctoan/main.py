# main.py
import os
import joblib
import warnings
import pandas as pd
import matplotlib.pyplot as plt

# Import các hàm từ 2 file vệ tinh kia
from data_processing import load_and_split_data, perform_eda, preprocess_pipeline, apply_undersampling
from modeling import train_model, evaluate_model

# Cấu hình chung
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
plt.style.use('ggplot')

def main():
    print("=== BẮT ĐẦU PIPELINE PHÁT HIỆN GIAN LẬN ===")
    
    # 1. Load Data
    X_train_raw, y_train, X_test_raw, y_test = load_and_split_data()
    
    # 2. EDA
    perform_eda(y_train)
    
    # 3. Preprocessing
    X_train_proc, X_test_proc = preprocess_pipeline(X_train_raw, X_test_raw)
    
    # 4. Undersampling
    X_train_res, y_train_res = apply_undersampling(X_train_proc, y_train)
    
    # 5. Training
    model = train_model(X_train_res, y_train_res, X_test_proc, y_test)
    
    # 6. Evaluation
    evaluate_model(model, X_test_proc, y_test)
    
    # Save Model
    if not os.path.exists('models'): os.makedirs('models')
    save_path = 'models/final_model_modular.pkl'
    joblib.dump(model, save_path)
    print(f"\n[XONG] Pipeline hoàn tất. Model lưu tại: {save_path}")

if __name__ == "__main__":
    main()