"""
GraphDTA Trainer

用于 GraphDTA 模型的训练，支持 DTI 分类和 DTA 回归任务。

特点:
1. 自适应损失函数：根据任务类型选择 CrossEntropyLoss 或 MSELoss
2. 支持多种评估指标
3. 与 opendrug pipeline 完全兼容
"""

import copy
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from trainer.BaseTrainer import BaseTrainer


class GraphDTA_Trainer(BaseTrainer):
    """
    GraphDTA 训练器

    支持:
    - DTI 分类任务: 预测药物-蛋白质是否存在相互作用
    - DTA 回归任务: 预测药物-蛋白质亲和力分数
    - 评估指标: Accuracy, Precision, Recall, F1, AUC, AP (分类) / MSE, MAE, RMSE, R2, Pearson, Spearman (回归)
    """

    def __init__(self, args, logger, dataset, model, optimizer):
        super().__init__(args, logger, dataset, model, optimizer)

        task_type = getattr(args, 'task', 'train_xxxx')
        self.task_type = 'regression' if task_type == 'dta' else 'classification'

        if self.task_type == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()

        self.best_model_state = None
        self.best_val_loss = float('inf')

    def _get_loss_function(self):
        return self.criterion

    def _prepare_batch_data(self, inp, task_type=None):
        """
        准备批次数据
        """
        labels = inp[2]
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        if self.task_type == 'classification':
            labels_out = labels.long()
        else:
            labels_out = labels.float()
        if self.device.type == 'cuda':
            labels_out = labels_out.to(self.device)
        return labels_out

    def _compute_classification_metrics(self, y_true, y_pred, y_prob=None):
        """计算分类评估指标"""
        metrics = {}

        if len(y_true) == 0 or len(y_pred) == 0:
            metrics['Accuracy'] = 0.0
            metrics['Precision'] = 0.0
            metrics['Recall'] = 0.0
            metrics['F1'] = 0.0
            metrics['AUC'] = 0.0
            metrics['AP'] = 0.0
            return metrics

        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        metrics['Precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['Recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['F1'] = f1_score(y_true, y_pred, average='binary', zero_division=0)

        if y_prob is not None and len(y_prob) > 0:
            try:
                metrics['AUC'] = roc_auc_score(y_true, y_prob)
                metrics['AP'] = average_precision_score(y_true, y_prob)
            except ValueError:
                metrics['AUC'] = 0.0
                metrics['AP'] = 0.0
        else:
            metrics['AUC'] = 0.0
            metrics['AP'] = 0.0

        return metrics

    def _compute_regression_metrics(self, y_true, y_pred):
        """计算回归评估指标"""
        metrics = {}

        if len(y_true) == 0 or len(y_pred) == 0:
            metrics['MSE'] = float('inf')
            metrics['RMSE'] = float('inf')
            metrics['MAE'] = float('inf')
            metrics['R2'] = 0.0
            metrics['Pearson'] = 0.0
            metrics['Spearman'] = 0.0
            return metrics

        metrics['MSE'] = mean_squared_error(y_true, y_pred)
        metrics['RMSE'] = np.sqrt(metrics['MSE'])
        metrics['MAE'] = mean_absolute_error(y_true, y_pred)

        try:
            metrics['R2'] = r2_score(y_true, y_pred)
        except ValueError:
            metrics['R2'] = 0.0

        try:
            pearson, _ = pearsonr(y_true, y_pred)
            metrics['Pearson'] = pearson if not np.isnan(pearson) else 0.0
        except:
            metrics['Pearson'] = 0.0

        try:
            spearman, _ = spearmanr(y_true, y_pred)
            metrics['Spearman'] = spearman if not np.isnan(spearman) else 0.0
        except:
            metrics['Spearman'] = 0.0

        return metrics

    def _train_epoch(self, epoch):
        """训练一个 epoch"""
        self.model.train()
        epoch_loss_sum = 0.0
        epoch_batches = 0
        ys, ypreds, yprobs = [], [], []

        for inp in self.dataset.train_loader:
            labels = self._prepare_batch_data(inp)

            self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(self.dataset.data_o, inp)

            if self.task_type == 'classification':
                loss = self.criterion(logits, labels)
            else:
                loss = self.criterion(logits.squeeze(), labels)

            loss.backward()
            self.optimizer.step()

            epoch_loss_sum += float(loss.item())
            epoch_batches += 1

            if self.task_type == 'classification':
                preds = logits.argmax(dim=1).detach().cpu().numpy()
                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            else:
                preds = logits.squeeze().detach().cpu().numpy()
                probs = preds

            ypreds.append(preds)
            yprobs.append(probs)
            ys.append(labels.detach().cpu().numpy())

        y_true = np.concatenate(ys) if ys else np.array([])
        y_pred = np.concatenate(ypreds) if ypreds else np.array([])
        y_prob = np.concatenate(yprobs) if yprobs else np.array([])

        if self.task_type == 'classification':
            metrics = self._compute_classification_metrics(y_true, y_pred, y_prob)
        else:
            metrics = self._compute_regression_metrics(y_true, y_pred)

        avg_loss = epoch_loss_sum / max(epoch_batches, 1)

        return metrics, avg_loss

    def _evaluate(self, loader):
        """评估模型"""
        self.model.eval()
        ys, ylogits, loss_sum, n = [], [], 0.0, 0

        with torch.no_grad():
            for inp in loader:
                labels = self._prepare_batch_data(inp)

                logits = self.model(self.dataset.data_o, inp)

                if self.task_type == 'classification':
                    loss = self.criterion(logits, labels)
                else:
                    loss = self.criterion(logits.squeeze(), labels)

                loss_sum += float(loss.item())
                n += 1

                ylogits.append(logits.detach().cpu())
                ys.append(labels.detach().cpu().numpy())

        y_true = np.concatenate(ys) if ys else np.array([])
        y_logits = torch.cat(ylogits, dim=0) if ylogits else torch.tensor([])

        if self.task_type == 'classification':
            y_pred_np = y_logits.argmax(dim=1).numpy()
            y_prob_np = torch.softmax(y_logits, dim=1)[:, 1].numpy()
            metrics = self._compute_classification_metrics(y_true, y_pred_np, y_prob_np)
        else:
            y_pred_np = y_logits.squeeze(-1).numpy()
            metrics = self._compute_regression_metrics(y_true, y_pred_np)

        avg_loss = loss_sum / max(n, 1)

        return metrics, avg_loss

    def train(self):
        """执行完整的训练流程"""
        task_name = "DTI 分类" if self.task_type == 'classification' else "DTA 回归"
        print("=" * 60)
        print(f"开始 GraphDTA 训练 ({task_name})")
        print("=" * 60)

        self._move_data_to_device()

        epochs = int(getattr(self.args, 'epochs', 150))
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            train_metrics, train_loss = self._train_epoch(epoch)
            val_metrics, val_loss = self._evaluate(self.dataset.val_loader)
            test_metrics, test_loss = self._evaluate(self.dataset.test_loader)

            if epoch % 10 == 0 or epoch == 1:
                print(f"\nEpoch {epoch}/{epochs}")
                print(f"  Train - Loss: {train_loss:.4f}")

                if self.task_type == 'classification':
                    print(f"  Train - Acc: {train_metrics['Accuracy']:.4f}, "
                          f"F1: {train_metrics['F1']:.4f}, AUC: {train_metrics['AUC']:.4f}")
                    print(f"  Val   - Acc: {val_metrics['Accuracy']:.4f}, "
                          f"F1: {val_metrics['F1']:.4f}, AUC: {val_metrics['AUC']:.4f}")
                    print(f"  Test  - Acc: {test_metrics['Accuracy']:.4f}, Prec: {test_metrics['Precision']:.4f}, "
                          f"Recall: {test_metrics['Recall']:.4f}, F1: {test_metrics['F1']:.4f}, "
                          f"AUC: {test_metrics['AUC']:.4f}, AP: {test_metrics['AP']:.4f}")
                else:
                    print(f"  Train - MSE: {train_metrics['MSE']:.4f}, "
                          f"RMSE: {train_metrics['RMSE']:.4f}, R2: {train_metrics['R2']:.4f}")
                    print(f"  Val   - MSE: {val_metrics['MSE']:.4f}, "
                          f"RMSE: {val_metrics['RMSE']:.4f}, R2: {val_metrics['R2']:.4f}")
                    print(f"  Test  - MSE: {test_metrics['MSE']:.4f}, RMSE: {test_metrics['RMSE']:.4f}, "
                          f"MAE: {test_metrics['MAE']:.4f}, R2: {test_metrics['R2']:.4f}, "
                          f"Pearson: {test_metrics['Pearson']:.4f}, Spearman: {test_metrics['Spearman']:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\n已加载最佳模型 (Epoch {best_epoch}, Val Loss: {self.best_val_loss:.4f})")

        final_metrics, _ = self._evaluate(self.dataset.test_loader)

        print("\n" + "=" * 60)
        print(f"GraphDTA 训练完成 ({task_name}) - 最终测试结果")
        print("=" * 60)

        if self.task_type == 'classification':
            print(f"  Accuracy:  {final_metrics['Accuracy']:.4f}")
            print(f"  Precision: {final_metrics['Precision']:.4f}")
            print(f"  Recall:    {final_metrics['Recall']:.4f}")
            print(f"  F1:        {final_metrics['F1']:.4f}")
            print(f"  AUC:       {final_metrics['AUC']:.4f}")
            print(f"  AP:        {final_metrics['AP']:.4f}")
        else:
            print(f"  MSE:      {final_metrics['MSE']:.4f}")
            print(f"  RMSE:     {final_metrics['RMSE']:.4f}")
            print(f"  MAE:      {final_metrics['MAE']:.4f}")
            print(f"  R2:       {final_metrics['R2']:.4f}")
            print(f"  Pearson:  {final_metrics['Pearson']:.4f}")
            print(f"  Spearman: {final_metrics['Spearman']:.4f}")

        self._save_results(final_metrics, best_epoch)

        return final_metrics

    def _save_results(self, final_metrics, best_epoch):
        """保存训练结果"""
        out_file = getattr(self.args, 'out_file', 'results.txt')
        # 文件名格式: model_task_dataset_results.txt (已在 parms_setting 中设置)
        # 如果 task_type 存在，替换 _results 为 _tasktype_results
        if self.task_type:
            out_file = out_file.replace('_results.txt', f'_{self.task_type}_results.txt')

        os.makedirs(os.path.dirname(out_file) if os.path.dirname(out_file) else '.', exist_ok=True)

        elapsed_time = self._get_elapsed_time()
        memory_usage = self._get_memory_usage()

        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 写入详细配置信息
        with open(out_file, 'a') as f:
            f.write("=" * 70 + "\n")
            f.write(f"Run at: {timestamp}\n")
            f.write("=" * 70 + "\n")
            f.write(f"Config Info:\n")
            f.write(f"  model: {getattr(self.args, 'model', 'GraphDTA')}\n")
            f.write(f"  task: {getattr(self.args, 'task', 'dti')}\n")
            f.write(f"  dataset: {getattr(self.args, 'matrix', 'dti')}\n")
            f.write(f"  modality: {getattr(self.args, 'modality', [])}\n")
            f.write(f"  noise_std: {getattr(self.args, 'noise_std', 0.0)}\n")
            f.write(f"  noise_ratio: {getattr(self.args, 'noise_ratio', 0.0)}\n")
            f.write(f"  noise_type: {getattr(self.args, 'noise_type', 'symmetric')}\n")
            f.write(f"  noise_edge: {getattr(self.args, 'noise_edge', 0.0)}\n")
            f.write(f"  sparse_drop_rate: {getattr(self.args, 'sparse_drop_rate', 0.0)}\n")
            f.write(f"  sparse_sample_rate: {getattr(self.args, 'sparse_sample_rate', 0.0)}\n\n")

            f.write(f"Results ({self.task_type}):\n")
            f.write(f"  Best Epoch: {best_epoch}\n")
            f.write(f"  Time: {elapsed_time:.3f}s\n")
            f.write(f"  GPU Memory: {memory_usage:.2f} MB\n\n")
            f.write(f"Final Metrics:\n")
            for k, v in final_metrics.items():
                if isinstance(v, float):
                    f.write(f"    {k}: {v:.6f}\n")
                else:
                    f.write(f"    {k}: {v}\n")
            f.write("\n")

        print(f"\n结果已保存到: {out_file}")
