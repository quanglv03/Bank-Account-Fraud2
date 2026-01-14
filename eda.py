import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")

# ===============================
# ĐƯỜNG DẪN & CẤU HÌNH
# ===============================
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = "eda_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ===============================
# HÀM HỖ TRỢ: TẠO CỜ NHỊ PHÂN (Cho EDA)
# ===============================
def create_behavior_flags(df):
    df = df.copy()

    df["bot_behavior"] = (df["keep_alive_session"] == 0).astype(int)
    df["repeat_device"] = (df["device_fraud_count"] > 0).astype(int)
    df["multi_email_device"] = (df["device_distinct_emails_8w"] >= 3).astype(int)
    
    # Tốc độ giao dịch cao
    if "velocity_6h" in df.columns and "velocity_24h" in df.columns:
        df["high_velocity"] = (
            (df["velocity_6h"] >= df["velocity_6h"].quantile(0.9)) |
            (df["velocity_24h"] >= df["velocity_24h"].quantile(0.9))
        ).astype(int)
    
    # Địa chỉ mới
    if "current_address_months_count" in df.columns and "prev_address_months_count" in df.columns:
        df["new_address"] = (
            (df["current_address_months_count"] < 6) |
            (df["prev_address_months_count"] < 2)
        ).astype(int)
        
    df["thin_credit"] = (df["has_other_cards"] == 0).astype(int)

    return df

# ===============================
# 1. PHÂN TÍCH HÀNH VI ĐƠN LẺ
# ===============================
def plot_distribution(df, col, bins=30):
    plt.figure(figsize=(7,4))
    sns.histplot(df[df["fraud_bool"] == 0][col], bins=bins, stat="density", label="Không gian lận", color="steelblue", alpha=0.5)
    sns.histplot(df[df["fraud_bool"] == 1][col], bins=bins, stat="density", label="Gian lận", color="salmon", alpha=0.6)
    plt.title(f"{col} – Gian lận vs Không gian lận")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{col}_distribution.png")
    plt.close()

def plot_fraud_rate(df, col, bins=None):
    data = df.copy()
    if bins is not None:
        data[col] = pd.cut(data[col], bins=bins)
    
    fraud_rate = data.groupby(col)["fraud_bool"].mean()
    plt.figure(figsize=(6,4))
    fraud_rate.plot(kind="bar", color="salmon")
    plt.title(f"Tỷ lệ gian lận theo {col}")
    plt.ylabel("Tỷ lệ gian lận")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{col}_fraud_rate.png")
    plt.close()

def analyze_fraud_behaviors(df):
    print("Đang chạy phân tích hành vi đơn lẻ...")
    # 1. BOT / CÔNG CỤ TỰ ĐỘNG
    plot_fraud_rate(df, "keep_alive_session")

    # 2. THIẾT BỊ
    plot_fraud_rate(df, "device_fraud_count")
    plot_fraud_rate(df, "device_distinct_emails_8w", bins=[0,1,2,3,5,10])

    # 3. TỐC ĐỘ GIAO DỊCH
    plot_distribution(df, "velocity_6h")
    plot_distribution(df, "velocity_24h")

    # 4. PHIÊN LÀM VIỆC
    plot_distribution(df, "session_length_in_minutes")
    plot_fraud_rate(df, "session_length_in_minutes", bins=[0,1,2,5,10,30])

    # 5. ĐỊA CHỈ
    plot_fraud_rate(df, "current_address_months_count", bins=[0,3,6,12,24,60])
    plot_fraud_rate(df, "prev_address_months_count", bins=[0,1,2,6,12])

    # 6. TÍN DỤNG
    plot_fraud_rate(df, "has_other_cards")
    plot_distribution(df, "credit_risk_score")
    plot_fraud_rate(df, "credit_risk_score", bins=[0,157,300,500,700])

# ===============================
# 2. PHÂN TÍCH KẾT HỢP HÀNH VI
# ===============================
def analyze_specific_combinations(df):
    print("Đang chạy phân tích kết hợp hành vi...")
    df = create_behavior_flags(df)
    combos = {
        "Bot + Địa chỉ mới": (df["bot_behavior"] == 1) & (df["new_address"] == 1),
        "Bot + Tín dụng thấp": (df["bot_behavior"] == 1) & (df["credit_risk_score"] <= 157),
        "Thiết bị lặp lại + Tốc độ cao": (df["repeat_device"] == 1) & (df["high_velocity"] == 1),
        "Nhiều Email + Bot": (df["multi_email_device"] == 1) & (df["bot_behavior"] == 1),
        "Tín dụng mỏng + Địa chỉ ổn định": (df["thin_credit"] == 1) & (df["current_address_months_count"] > 46),
    }

    results = []
    for name, condition in combos.items():
        subset = df[condition]
        fraud_rate = subset["fraud_bool"].mean()
        count = len(subset)
        results.append({"Sự kết hợp": name, "Số mẫu": count, "Tỷ lệ gian lận": fraud_rate})

    result_df = pd.DataFrame(results).sort_values("Tỷ lệ gian lận", ascending=False)

    plt.figure(figsize=(8,4))
    plt.bar(result_df["Sự kết hợp"], result_df["Tỷ lệ gian lận"], color="salmon")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Tỷ lệ gian lận")
    plt.title("Tỷ lệ gian lận của các hành vi kết hợp")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/combo_fraud_rate.png")
    plt.close()
    
    result_df.to_csv(f"{OUT_DIR}/combo_summary.csv", index=False)
    return result_df

def plot_behavior_cooccurrence(df):
    df = create_behavior_flags(df)
    behavior_cols = ["bot_behavior", "repeat_device", "multi_email_device", "high_velocity", "new_address", "thin_credit"]
    fraud_df = df[df["fraud_bool"] == 1][behavior_cols]
    corr = fraud_df.corr()

    plt.figure(figsize=(6,5))
    sns.heatmap(corr, annot=True, cmap="Reds", fmt=".2f")
    plt.title("Đồng xuất hiện hành vi (Chỉ gian lận)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/behavior_cooccurrence_heatmap.png")
    plt.close()

# ===============================
# 3. PHÂN TÍCH SO SÁNH
# ===============================
def compare_single_vs_combo(df):
    print("Đang chạy phân tích so sánh...")
    df = create_behavior_flags(df)
    base_rate = df["fraud_bool"].mean()

    scenarios = {
        "Chỉ Bot": df["bot_behavior"] == 1,
        "Chỉ tốc độ cao": df["high_velocity"] == 1,
        "Chỉ thiết bị lặp lại": df["repeat_device"] == 1,
        "Bot + Địa chỉ mới": (df["bot_behavior"] == 1) & (df["new_address"] == 1),
        "Thiết bị lặp lại + Tốc độ cao": (df["repeat_device"] == 1) & (df["high_velocity"] == 1),
        "Bot + Tín dụng mỏng": (df["bot_behavior"] == 1) & (df["thin_credit"] == 1),
    }

    results = []
    for name, cond in scenarios.items():
        subset = df[cond]
        fraud_rate = subset["fraud_bool"].mean()
        count = len(subset)
        results.append({
            "Kịch bản": name, 
            "Số mẫu": count, 
            "Tỷ lệ gian lận": fraud_rate,
            "Tăng so với tổng thể": fraud_rate / base_rate if base_rate > 0 else 0
        })

    result_df = pd.DataFrame(results).sort_values("Tỷ lệ gian lận", ascending=False)

    plt.figure(figsize=(9,4))
    plt.bar(result_df["Kịch bản"], result_df["Tỷ lệ gian lận"], color="salmon")
    plt.axhline(base_rate, color="black", linestyle="--", label="Tỷ lệ gian lận tổng thể")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Tỷ lệ gian lận")
    plt.title("Hành vi gian lận Đơn lẻ vs Kết hợp")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/single_vs_combo_fraud_rate.png")
    plt.close()

    result_df.to_csv(f"{OUT_DIR}/single_vs_combo_summary.csv", index=False)
    return result_df

# ===============================
# MAIN
# ===============================
def run_all_eda(df):
    analyze_fraud_behaviors(df)
    analyze_specific_combinations(df)
    plot_behavior_cooccurrence(df)
    compare_single_vs_combo(df)
    print(f"Tất cả các tác vụ EDA đã hoàn thành. Kết quả được lưu tại {OUT_DIR}")

if __name__ == "__main__":
    # Dành cho mục đích kiểm thử
    DATA_PATH = os.path.join(BASE_PATH, "..", "data", "Base.csv")
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        run_all_eda(df)
    else:
        print("Không tìm thấy dữ liệu.")
