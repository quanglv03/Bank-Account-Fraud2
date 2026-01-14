# Hướng dẫn cài đặt và chạy dự án Bank_2

## 1. Giới thiệu

- Dự án phát hiện gian lận giao dịch ngân hàng.
- Gồm nhiều pipeline mô hình (CatBoost, LightGBM, XGBoost) trong thư mục `src`.

## 2. Yêu cầu môi trường

- Python 3.11 trở lên (khuyến nghị dùng phiên bản gần với 3.13).
- `pip` để cài đặt thư viện.

## 3. Cài đặt thư viện

Từ thư mục chứa `Bank_2`:

```bash
cd PROJECT_NHÓM 4/PROJECT_NHÓM 4/Bank_2
pip install -r requirements.txt
```

Nếu thiếu thư viện bổ sung (ví dụ: `joblib`, `seaborn`, `imblearn`), cài thêm bằng:

```bash
pip install joblib seaborn imbalanced-learn
```

## 4. Dữ liệu

- File dữ liệu chính: `data/Base.csv`.
- Đảm bảo file này tồn tại đúng vị trí trước khi chạy các pipeline.

## 5. Cách chạy các pipeline

### 5.1. Chạy pipeline CatBoost (catboost_levanquang)

Từ thư mục gốc chứa thư mục `Bank_2` (tức thư mục có cấu trúc `Bank_2/src/...`):

```bash
python -m Bank_2.src.catboost_levanquang.main
```

Pipeline này sẽ:

- Tải và tiền xử lý dữ liệu.
- Chạy EDA và lưu kết quả vào thư mục `eda_results/`.
- Huấn luyện mô hình CatBoost và đánh giá, tìm ngưỡng tối ưu.

### 5.2. Chạy pipeline LightGBM (lightgbm_nguyenphuoctoan)

Từ thư mục `Bank_2`:

```bash
cd PROJECT_NHÓM 4/PROJECT_NHÓM 4/Bank_2
python src/lightgbm_nguyenphuoctoan/main.py
```

Pipeline này sẽ:

- Tự tìm file `Base.csv` trong cây thư mục.
- Thực hiện EDA, tiền xử lý, cân bằng dữ liệu và huấn luyện mô hình LightGBM.
- Lưu mô hình đã huấn luyện vào thư mục `models/`.

### 5.3. Chạy pipeline XGBoost (xgboost_nguenhoangtuananh)

Mặc định, file `main.py` đang sử dụng biến `BASE_DIR` trỏ tới đường dẫn dữ liệu trên máy cá nhân.  
Trước khi chạy, cần sửa `BASE_DIR` trong file:

- Mở file: `src/xgboost_nguenhoangtuananh/main.py`.
- Cập nhật `BASE_DIR` thành đường dẫn thư mục chứa `Base.csv` trên máy của bạn.

Sau đó, từ thư mục `Bank_2`:

```bash
cd PROJECT_NHÓM 4/PROJECT_NHÓM 4/Bank_2
python src/xgboost_nguenhoangtuananh/main.py
```

## 6. Gợi ý quy trình làm việc

1. Kiểm tra dữ liệu đã nằm đúng thư mục `data/Base.csv`.
2. Cài đặt đủ thư viện với `pip install -r requirements.txt`.
3. Chạy pipeline CatBoost trước để xem toàn bộ luồng (EDA → huấn luyện → đánh giá).
4. Sau đó có thể thử thêm LightGBM và XGBoost để so sánh kết quả.

## 7. Ghi chú

- Nếu gặp lỗi `ModuleNotFoundError`, hãy kiểm tra lại:
  - Thư mục hiện tại khi chạy lệnh.
  - Đường dẫn dữ liệu trong các file `main.py`.
- Có thể tạo môi trường ảo riêng (venv, conda) để tránh xung đột thư viện.
