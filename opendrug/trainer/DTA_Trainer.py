"""
DTA (Drug-Target Affinity) Trainer
用于药物-蛋白质亲和力预测任务的训练器

与 DDI Trainer 的主要区别:
1. 损失函数: MSELoss (回归) vs CrossEntropyLoss (分类)
2. 评估指标: MSE, RMSE, MAE, Pearson, Spearman vs Accuracy, F1
3. 数据格式: (drug_idx, protein_idx, affinity) vs (drug1_idx, drug2_idx, relation_type)
"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple, Any
from trainer.BaseTrainer import BaseTrainer


class DTA_Trainer(BaseTrainer):
    """
    DTA 训练器

    支持:
    - 回归任务: 预测药物-蛋白质亲和力分数
    - 评估指标: MSE, RMSE, MAE, Pearson, Spearman
    """

    def __init__(self, args, logger, dataset, model, optimizer):
        """
        初始化 DTA 训练器

        Args:
            args: 配置参数
            logger: 日志记录器
            dataset: 数据集对象
            model: 神经网络模型
            optimizer: 优化器
        """
        super().__init__(args, logger, dataset, model, optimizer)

        # 回归任务的损失函数
        self.criterion = nn.MSELoss()

        # 最佳模型追踪
        self.best_model_state = None
        self.best_val_loss = float('inf')

    def _get_loss_function(self):
        """获取损失函数"""
        return self.criterion

    def _prepare_batch_data(self, inp, task_type='dta'):
        """
        准备批次数据

        Args:
            inp: 输入数据 (drug_idx, protein_idx, label)
            task_type: 任务类型

        Returns:
            labels: 亲和力标签
        """
        labels = inp[2]
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        labels = labels.float()
        if self.device.type == 'cuda':
            labels = labels.cuda()
        return labels

    def _compute_metrics(self, y_true, y_pred):
        """
        计算评估指标

        Args:
            y_true: 真实值
            y_pred: 预测值

        Returns:
            metrics: 指标字典
        """
        metrics = {}

        # MSE 和 RMSE
        metrics['MSE'] = mean_squared_error(y_true, y_pred)
        metrics['RMSE'] = np.sqrt(metrics['MSE'])

        # MAE
        metrics['MAE'] = mean_absolute_error(y_true, y_pred)

        # R² Score
        metrics['R2'] = r2_score(y_true, y_pred)

        # Pearson 相关系数
        try:
            pearson, _ = pearsonr(y_true, y_pred)
            metrics['Pearson'] = pearson if not np.isnan(pearson) else 0.0
        except:
            metrics['Pearson'] = 0.0

        # Spearman 相关系数
        try:
            spearman, _ = spearmanr(y_true, y_pred)
            metrics['Spearman'] = spearman if not np.isnan(spearman) else 0.0
        except:
            metrics['Spearman'] = 0.0

        return metrics

    def _train_epoch(self, epoch, task_type='dta'):
        """
        训练一个 epoch

        Args:
            epoch: 当前 epoch
            task_type: 任务类型

        Returns:
            metrics: 训练指标
            avg_loss: 平均损失
        """
        self.model.train()
        epoch_loss_sum = 0.0
        epoch_batches = 0
        ys, ypreds = [], []

        for inp in self.dataset.train_loader:
            labels = self._prepare_batch_data(inp, task_type)

            self.optimizer.zero_grad(set_to_none=True)

            # 前向传播
            logits = self.model(self.dataset.data_o, inp)
            loss = self.criterion(logits.squeeze(), labels)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            epoch_loss_sum += float(loss.item())
            epoch_batches += 1

            # 收集预测结果
            preds = logits.detach().cpu().squeeze().numpy()
            if preds.ndim == 0:
                preds = np.array([preds.item()])
            ypreds.append(preds)
            ys.append(labels.detach().cpu().numpy())

        # 合并所有批次的数据
        y_true = np.concatenate(ys) if ys else np.array([])
        y_pred = np.concatenate(ypreds) if ypreds else np.array([])

        # 计算指标
        metrics = self._compute_metrics(y_true, y_pred)
        avg_loss = epoch_loss_sum / max(epoch_batches, 1)

        return metrics, avg_loss

    def _evaluate(self, loader, task_type='dta'):
        """
        评估模型

        Args:
            loader: 数据加载器
            task_type: 任务类型

        Returns:
            metrics: 评估指标
            avg_loss: 平均损失
        """
        self.model.eval()
        ys, ylogits, loss_sum, n = [], [], 0.0, 0

        with torch.no_grad():
            for inp in loader:
                labels = self._prepare_batch_data(inp, task_type)

                logits = self.model(self.dataset.data_o, inp)
                loss = self.criterion(logits.squeeze(), labels)

                loss_sum += float(loss.item())
                n += 1

                preds = logits.detach().cpu().squeeze().numpy()
                if preds.ndim == 0:
                    preds = np.array([preds.item()])
                ylogits.append(preds)
                ys.append(labels.detach().cpu().numpy())

        y_true = np.concatenate(ys) if ys else np.array([])
        y_pred = np.concatenate(ylogits) if ylogits else np.array([])

        metrics = self._compute_metrics(y_true, y_pred)
        avg_loss = loss_sum / max(n, 1)

        return metrics, avg_loss

    def train(self):
        """执行完整的训练流程"""
        print("=" * 60)
        print("开始 DTA 训练")
        print("=" * 60)

        self._move_data_to_device()

        epochs = int(getattr(self.args, 'epochs', 150))
        best_metrics = None
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            # 训练
            train_metrics, train_loss = self._train_epoch(epoch)

            # 验证
            val_metrics, val_loss = self._evaluate(self.dataset.val_loader)

            # 测试
            test_metrics, test_loss = self._evaluate(self.dataset.test_loader)

            # 打印信息
            if epoch % 10 == 0 or epoch == 1:
                print(f"\nEpoch {epoch}/{epochs}")
                print(f"  Train - Loss: {train_loss:.4f}, MSE: {train_metrics['MSE']:.4f}, "
                      f"RMSE: {train_metrics['RMSE']:.4f}, MAE: {train_metrics['MAE']:.4f}")
                print(f"  Val   - Loss: {val_loss:.4f}, MSE: {val_metrics['MSE']:.4f}, "
                      f"RMSE: {val_metrics['RMSE']:.4f}, MAE: {val_metrics['MAE']:.4f}")
                print(f"  Test  - MSE: {test_metrics['MSE']:.4f}, RMSE: {test_metrics['RMSE']:.4f}, "
                      f"MAE: {test_metrics['MAE']:.4f}, Pearson: {test_metrics['Pearson']:.4f}, "
                      f"Spearman: {test_metrics['Spearman']:.4f}")

            # 保存最佳模型（基于验证集 MSE）
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                best_metrics = test_metrics
                best_epoch = epoch

        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\n已加载最佳模型 (Epoch {best_epoch}, Val MSE: {self.best_val_loss:.4f})")

        # 最终评估
        final_metrics, _ = self._evaluate(self.dataset.test_loader)

        print("\n" + "=" * 60)
        print("DTA 训练完成 - 最终测试结果")
        print("=" * 60)
        print(f"  MSE:     {final_metrics['MSE']:.4f}")
        print(f"  RMSE:    {final_metrics['RMSE']:.4f}")
        print(f"  MAE:     {final_metrics['MAE']:.4f}")
        print(f"  R2:      {final_metrics['R2']:.4f}")
        print(f"  Pearson: {final_metrics['Pearson']:.4f}")
        print(f"  Spearman: {final_metrics['Spearman']:.4f}")

        # 保存结果
        self._save_results(final_metrics, best_epoch)

        return final_metrics

    def _save_results(self, final_metrics, best_epoch):
        """保存训练结果"""
        out_file = getattr(self.args, 'out_file', 'results.txt')
        # 文件名格式: model_task_dataset_results.txt (已在 parms_setting 中设置)

        os.makedirs(os.path.dirname(out_file), exist_ok=True)

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
            f.write(f"  model: {getattr(self.args, 'model', 'DTA')}\n")
            f.write(f"  task: {getattr(self.args, 'task', 'dta')}\n")
            f.write(f"  dataset: {getattr(self.args, 'matrix', 'dta')}\n")
            f.write(f"  modality: {getattr(self.args, 'modality', [])}\n")
            f.write(f"  noise_std: {getattr(self.args, 'noise_std', 0.0)}\n")
            f.write(f"  noise_ratio: {getattr(self.args, 'noise_ratio', 0.0)}\n")
            f.write(f"  noise_type: {getattr(self.args, 'noise_type', 'symmetric')}\n")
            f.write(f"  noise_edge: {getattr(self.args, 'noise_edge', 0.0)}\n")
            f.write(f"  sparse_drop_rate: {getattr(self.args, 'sparse_drop_rate', 0.0)}\n")
            f.write(f"  sparse_sample_rate: {getattr(self.args, 'sparse_sample_rate', 0.0)}\n\n")

            f.write(f"Results (regression):\n")
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


import os
