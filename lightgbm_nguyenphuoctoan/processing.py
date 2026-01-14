# data_processing.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from collections import Counter

from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler

def load_and_split_data():
    print("\n[BƯỚC 1] TẢI DỮ LIỆU...")
    matches = glob.glob(os.path.join("**", "Base.csv"), recursive=True)
    if not matches:
        all_csv = glob.glob("**/*.csv", recursive=True)
        if all_csv: 
            path = all_csv[0]
        else:
            raise FileNotFoundError("Lỗi: Không tìm thấy file csv nào!")
    else:
        path = matches[0]
        
    print(f"   -> Đang đọc file từ đường dẫn: {path}")
    df = pd.read_csv(path)
    
    print("[BƯỚC 1.1] CHIA TẬP DỮ LIỆU THEO THỜI GIAN (TEMPORAL SPLIT)...")
    train_mask = df['month'] < 6
    
    X_train = df[train_mask].drop(['fraud_bool', 'month'], axis=1)
    y_train = df[train_mask]['fraud_bool']
    
    X_test = df[~train_mask].drop(['fraud_bool', 'month'], axis=1)
    y_test = df[~train_mask]['fraud_bool']
    
    print(f"   -> Train: {X_train.shape} | Fraud Rate: {y_train.mean():.2%}")
    print(f"   -> Test : {X_test.shape} | Fraud Rate: {y_test.mean():.2%}")
    
    return X_train, y_train, X_test, y_test

def perform_eda(y_train):
    print("\n[BƯỚC 2] PHÂN TÍCH DỮ LIỆU SƠ BỘ (EDA)...")
    plt.figure(figsize=(8, 4))
    ax = sns.countplot(x=y_train, palette='viridis')
    plt.title(f'Phân phối nhãn (Tổng: {len(y_train)})')
    plt.xlabel('Nhãn (0: Bình thường, 1: Gian lận)')
    
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    plt.show()

def preprocess_pipeline(X_train, X_test):
    print("\n[BƯỚC 3] TIỀN XỬ LÝ DỮ LIỆU...")
    
    # 3.1. Missing Values
    print("   -> [3.1] Xử lý giá trị thiếu...")
    missing_cols = ['prev_address_months_count', 'current_address_months_count', 
                    'bank_months_count', 'session_length_in_minutes', 'device_distinct_emails_8w']
    for col in missing_cols:
        if col in X_train.columns:
            median_val = X_train[X_train[col] != -1][col].median()
            X_train.loc[X_train[col] == -1, col] = median_val
            X_test.loc[X_test[col] == -1, col] = median_val

    # 3.2. Feature Engineering
    print("   -> [3.2] Tạo đặc trưng mới...")
    def create_features(df):
        df = df.copy()
        denom = df['session_length_in_minutes'].replace(0, 0.1) + 1
        df['velocity'] = df['intended_balcon_amount'] / denom
        df['log_amount'] = np.log1p(np.abs(df['intended_balcon_amount']))
        denom_bank = df['bank_months_count'].replace(-1, 0) + 1
        df['age_bank_interact'] = df['customer_age'] * 12 / denom_bank
        return df
    
    X_train = create_features(X_train)
    X_test = create_features(X_test)
    
    # 3.3. Encoding & Scaling
    print("   -> [3.3] Mã hóa & Chuẩn hóa...")
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    num_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()
    
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    X_train_cat = pd.DataFrame(ohe.fit_transform(X_train[cat_cols]), index=X_train.index)
    X_test_cat = pd.DataFrame(ohe.transform(X_test[cat_cols]), index=X_test.index)
    
    scaler = RobustScaler()
    X_train_num = pd.DataFrame(scaler.fit_transform(X_train[num_cols]), columns=num_cols, index=X_train.index)
    X_test_num = pd.DataFrame(scaler.transform(X_test[num_cols]), columns=num_cols, index=X_test.index)
    
    X_train_final = pd.concat([X_train_num, X_train_cat], axis=1)
    X_test_final = pd.concat([X_test_num, X_test_cat], axis=1)
    
    X_train_final.columns = X_train_final.columns.astype(str)
    X_test_final.columns = X_test_final.columns.astype(str)
    
    # 3.4. Feature Selection
    print("   -> [3.4] Loại bỏ cột rác...")
    selector = VarianceThreshold(threshold=0)
    selector.fit(X_train_final)
    cols_keep = X_train_final.columns[selector.get_support()]
    X_train_final = X_train_final[cols_keep]
    X_test_final = X_test_final[cols_keep]
    
    imputer = SimpleImputer(strategy='median')
    X_train_final = pd.DataFrame(imputer.fit_transform(X_train_final), columns=X_train_final.columns)
    X_test_final = pd.DataFrame(imputer.transform(X_test_final), columns=X_test_final.columns)
    
    print(f"   -> Kích thước sau xử lý: {X_train_final.shape}")
    return X_train_final, X_test_final

def apply_undersampling(X_train, y_train):
    print("\n[BƯỚC 4] CÂN BẰNG DỮ LIỆU (UNDERSAMPLING)...")
    print(f"   -> Trước khi giảm: {Counter(y_train)}")
    rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    print(f"   -> Sau khi giảm  : {Counter(y_res)}")
    return X_res, y_res