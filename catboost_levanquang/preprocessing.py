import pandas as pd
import os

def load_data(data_path=None):
    if data_path is None:
        # Đường dẫn mặc định tương đối với file này
        base_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_path, "..", "data", "Base.csv")
    
    print(f"Đang tải dữ liệu từ: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại {data_path}")
        
    df = pd.read_csv(data_path)
    return df

def preprocess_data(df):
    """
    Các bước tiền xử lý cơ bản như xử lý giá trị thiếu, v.v.
    Dataset hiện tại có vẻ sạch, nên bước này có thể tối thiểu.
    """
    # Chỗ để cho bất kỳ việc làm sạch cơ bản nào nếu cần
    # Hiện tại chỉ trả về bản sao
    return df.copy()
