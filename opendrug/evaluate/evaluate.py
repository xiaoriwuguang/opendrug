import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
    recall_score, precision_score, precision_recall_curve, auc
)
from matplotlib import pyplot as plt


def _softmax(logits: np.ndarray) -> np.ndarray:
    """对 logits 应用 softmax 转换为概率"""
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def _metrics_from_logits(y_true_np, y_logits_np, num_classes=None):
    """返回 Acc/F1/Recall/Precision/AUC-ROC（macro，且避免未定义警告）

    Args:
        y_true_np: 真实标签
        y_logits_np: 模型 logits
        num_classes: 类别总数（可选）
    """
    # 确保是 numpy 数组和正确类型
    y_logits_np = np.asarray(y_logits_np, dtype=np.float32)
    y_true_np = np.asarray(y_true_np, dtype=np.int64)

    y_pred_np = np.argmax(y_logits_np, axis=1)
    acc = accuracy_score(y_true_np, y_pred_np)
    f1  = f1_score(y_true_np, y_pred_np, average='macro',   zero_division=0)
    rec = recall_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    pre = precision_score(y_true_np, y_pred_np, average='macro', zero_division=0)

    # 计算 AUC-ROC (macro)
    try:
        from scipy.special import softmax
        y_prob = softmax(y_logits_np, axis=1)

        n_classes = y_prob.shape[1]
        n_classes_true = int(y_true_np.max()) + 1

        # 逐类别计算 AUC，然后取平均
        auc_scores = []
        for c in range(n_classes):
            # 创建二值标签：当前类别 vs 其他
            y_true_binary = (y_true_np == c).astype(int)
            # 检查是否有正负样本
            if y_true_binary.sum() > 0 and y_true_binary.sum() < len(y_true_binary):
                try:
                    auc_c = roc_auc_score(y_true_binary, y_prob[:, c])
                    auc_scores.append(auc_c)
                except ValueError:
                    pass  # 跳过无法计算的类别

        if auc_scores:
            auc_roc = np.mean(auc_scores)
        else:
            auc_roc = 0.0
    except Exception as e:
        print(f"[WARN] AUC-ROC 计算失败: {e}")
        auc_roc = 0.0
    return acc, f1, rec, pre, auc_roc

def _metrics_from_logits_multilabel(y_true_np, y_logits_np):
    """返回 AUC 和 AP（macro 平均，避免未定义警告）"""
    auc_macro = 0.0
    ap_macro = 0.0
    valid_k = y_true_np.shape[1]  # 标签数量
    for k in range(y_true_np.shape[1]):
        if np.sum(y_true_np[:, k]) < 1 or np.sum(y_true_np[:, k]) == len(y_true_np[:, k]):
            valid_k -= 1  # 跳过全0或全1的标签
            continue
        auc_macro += roc_auc_score(y_true_np[:, k], y_logits_np[:, k])
        ap_macro += average_precision_score(y_true_np[:, k], y_logits_np[:, k])
    auc_macro = auc_macro / valid_k if valid_k > 0 else 0.0
    ap_macro = ap_macro / valid_k if valid_k > 0 else 0.0
    return roc_auc_score(y_true_np.reshape(-1), y_logits_np.reshape(-1)), ap_macro

def plot_metrics(train_metrics, val_metrics, metric_name, out_file_prefix):
    epochs = range(1, len(train_metrics) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_metrics, 'b-', label=f'Train {metric_name}')
    plt.plot(epochs, val_metrics, 'r-', label=f'Val {metric_name}')
    plt.xlabel('Epoch'); plt.ylabel(metric_name)
    plt.title(f'{metric_name} vs. Epoch'); plt.legend(); plt.grid(True)
    plt.savefig(f'{out_file_prefix}_{metric_name.lower().replace(" ", "_")}.png')
    plt.close()