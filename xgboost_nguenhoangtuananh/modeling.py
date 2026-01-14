# modeling.py
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, plot_importance
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, precision_recall_curve
)

def train_xgboost(X_train, y_train, preprocessor):
    print("\n[BƯỚC 6] HUẤN LUYỆN XGBOOST...")
    
    # Cấu hình XGBoost như bạn yêu cầu
    xgb_clf = XGBClassifier(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.01,
        min_child_weight=1,
        gamma=1,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=20.0,   # Xử lý mất cân bằng
        tree_method="hist",      # Tối ưu tốc độ
        device="cuda",           # Dùng GPU (nếu có)
        eval_metric="logloss",
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb_clf)
    ])

    # Huấn luyện
    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        print(f"Lỗi khi training (có thể do thiếu GPU): {e}")
        print("Chuyển sang chế độ CPU...")
        xgb_clf.set_params(device="cpu")
        pipeline.set_params(classifier=xgb_clf)
        pipeline.fit(X_train, y_train)
        
    print("   -> Huấn luyện xong.")
    return pipeline

def find_best_threshold(y_test, y_prob):
    """Tìm ngưỡng tối ưu dựa trên F1-Score."""
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    print(f"   -> Best F1: {f1_scores[best_idx]:.4f}")
    print(f"   -> Best Threshold: {best_threshold:.4f}")
    return best_threshold

def evaluate_model(pipeline, X_test, y_test, best_threshold):
    print("\n[BƯỚC 7] ĐÁNH GIÁ MÔ HÌNH...")
    
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"   -> ROC-AUC Score: {roc_auc:.4f}")

    test_thresholds = [0.5, best_threshold]
    
    for th in test_thresholds:
        print(f"\n--- REPORT VỚI THRESHOLD = {th:.4f} ---")
        y_pred_th = (y_prob >= th).astype(int)
        
        print(classification_report(y_test, y_pred_th, digits=4))
        
        # Vẽ Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_th)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Fraud", "Fraud"])
        fig, ax = plt.subplots(figsize=(5, 5))
        disp.plot(values_format="d", ax=ax, cmap="Blues")
        plt.title(f"Confusion Matrix (Th={th:.4f})")
        plt.show()

def plot_feature_importance(pipeline):
    print("\n[BƯỚC 8] VẼ FEATURE IMPORTANCE...")
    model = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']
    
    try:
        feature_names = preprocessor.get_feature_names_out()
        model.get_booster().feature_names = list(feature_names)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        plot_importance(model, max_num_features=20, height=0.5, ax=ax, 
                        title="Feature Importance (XGBoost)", importance_type='weight')
        plt.show()
    except Exception as e:
        print(f"Không thể vẽ Feature Importance: {e}")