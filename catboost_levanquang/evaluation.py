from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
import pandas as pd

def evaluate_model(model, X_test, y_test, threshold=0.5):
    probs = model.predict_proba(X_test)[:, 1]
    
    # ngưỡng mặc định
    y_pred = (probs >= threshold).astype(int)

    print("\n=== MA TRẬN NHẦM LẪN ===")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\n=== BÁO CÁO PHÂN LOẠI ===")
    print(classification_report(y_test, y_pred, digits=4))

    roc_auc = roc_auc_score(y_test, probs)
    print("ROC-AUC:", roc_auc)
    
    return {
        "confusion_matrix": cm,
        "classification_report": classification_report(y_test, y_pred, digits=4, output_dict=True),
        "roc_auc": roc_auc
    }

def evaluate_at_threshold(y_true, probs, threshold):
    y_pred = (probs >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return cm, precision, recall, f1

def search_threshold(model, X_test, y_test, thresholds=[0.3, 0.4, 0.5]):
    probs = model.predict_proba(X_test)[:, 1]
    
    print("\n==============================")
    print("TÌM KIẾM NGƯỠNG (Lớp Gian Lận)")
    print("==============================")

    for t in thresholds:
        cm, p, r, f1 = evaluate_at_threshold(y_test, probs, t)

        print(f"\n--- Ngưỡng = {t} ---")
        print("Ma trận nhầm lẫn:")
        print(cm)
        print(f"Độ chính xác (Precision): {p:.4f} | Độ nhạy (Recall): {r:.4f} | F1: {f1:.4f}")
