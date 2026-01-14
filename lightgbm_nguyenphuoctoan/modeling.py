# modeling.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

def train_model(X_train, y_train, X_val, y_val):
    print("\n[BƯỚC 5] HUẤN LUYỆN MÔ HÌNH (RandomizedSearchCV)...")
    
    param_dist = {
        'num_leaves': [31, 50, 70],
        'max_depth': [10, 15, -1],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [500, 1000],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8],
    }
    
    lgb_base = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
    
    print("   -> Đang dò tìm tham số tối ưu...")
    random_search = RandomizedSearchCV(
        estimator=lgb_base,
        param_distributions=param_dist,
        n_iter=10,
        scoring='roc_auc',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    print(f"   -> Tham số tốt nhất: {random_search.best_params_}")
    
    print("   -> Huấn luyện model cuối cùng...")
    best_model = lgb.LGBMClassifier(**random_search.best_params_, random_state=42, n_jobs=-1, verbose=-1)
    
    best_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    return best_model

def evaluate_model(model, X_test, y_test):
    print("\n[BƯỚC 6] ĐÁNH GIÁ HIỆU NĂNG MÔ HÌNH...")
    
    y_prob = model.predict_proba(X_test)[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)
    best_threshold = thresholds[ix]
    
    print(f"   -> Ngưỡng tối ưu: {best_threshold:.4f}")
    print(f"   -> AUC Score: {auc(fpr, tpr):.4f}")
    
    y_pred = (y_prob >= best_threshold).astype(int)
    
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred, target_names=['Bình thường', 'Gian lận']))
    
    # Vẽ biểu đồ
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    # Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0])
    ax[0].set_title(f'Ma trận nhầm lẫn @ {best_threshold:.2f}')
    
    # ROC
    ax[1].plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}', color='red', lw=2)
    ax[1].scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best Threshold')
    ax[1].plot([0, 1], [0, 1], 'k--')
    ax[1].legend()
    
    plt.tight_layout()
    plt.show()