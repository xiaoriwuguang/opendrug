"""
DTI (Drug-Target Interaction) 分类 Trainer
用于药物-蛋白质相互作用分类预测任务的训练器

与 DDI Trainer 的主要区别:
1. 损失函数: CrossEntropyLoss (二分类)
2. 评估指标: Accuracy, Precision, Recall, F1, AUC
3. 数据格式: (drug_idx, protein_idx, label)
"""

import copy
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from typing import Dict, Tuple, Any
from trainer.BaseTrainer import BaseTrainer


class DTI_Trainer(BaseTrainer):
    """
    DTI 训练器

    支持:
    - 二分类任务: 预测药物-蛋白质是否存在相互作用
    - 评估指标: Accuracy, Precision, Recall, F1, AUC, AP
    """

    def __init__(self, args, logger, dataset, model, optimizer):
        super().__init__(args, logger, dataset, model, optimizer)

        self.criterion = nn.CrossEntropyLoss()
        self.best_model_state = None
        self.best_val_loss = float('inf')

    def _get_loss_function(self):
        return self.criterion

    def _prepare_batch_data(self, inp, task_type='dti'):
        labels = inp[2]
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        labels = labels.long()
        if self.device.type == 'cuda':
            labels = labels.cuda()
        return labels

    def _compute_metrics(self, y_true, y_pred, y_prob=None):
        metrics = {}

        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        metrics['Precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['Recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['F1'] = f1_score(y_true, y_pred, average='binary', zero_division=0)

        if y_prob is not None:
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

    def _train_epoch(self, epoch, task_type='dti'):
        self.model.train()
        epoch_loss_sum = 0.0
        epoch_batches = 0
        ys, ypreds, yprobs = [], [], []

        for inp in self.dataset.train_loader:
            labels = self._prepare_batch_data(inp, task_type)

            self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(self.dataset.data_o, inp)
            loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()

            epoch_loss_sum += float(loss.item())
            epoch_batches += 1

            preds = logits.argmax(dim=1).detach().cpu().numpy()
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            ypreds.append(preds)
            yprobs.append(probs)
            ys.append(labels.detach().cpu().numpy())

        y_true = np.concatenate(ys) if ys else np.array([])
        y_pred = np.concatenate(ypreds) if ypreds else np.array([])
        y_prob = np.concatenate(yprobs) if yprobs else np.array([])

        metrics = self._compute_metrics(y_true, y_pred, y_prob)
        avg_loss = epoch_loss_sum / max(epoch_batches, 1)

        return metrics, avg_loss

    def _evaluate(self, loader, task_type='dti'):
        self.model.eval()
        ys, ylogits, loss_sum, n = [], [], 0.0, 0

        with torch.no_grad():
            for inp in loader:
                labels = self._prepare_batch_data(inp, task_type)

                logits = self.model(self.dataset.data_o, inp)
                loss = self.criterion(logits, labels)

                loss_sum += float(loss.item())
                n += 1

                ylogits.append(logits.detach().cpu())
                ys.append(labels.detach().cpu().numpy())

        y_true = np.concatenate(ys) if ys else np.array([])
        y_logits = torch.cat(ylogits, dim=0) if ylogits else torch.tensor([])
        y_pred = y_logits.argmax(dim=1).numpy() if ylogits else np.array([])
        y_prob = torch.softmax(y_logits, dim=1)[:, 1].numpy() if ylogits else np.array([])

        metrics = self._compute_metrics(y_true, y_pred, y_prob)
        avg_loss = loss_sum / max(n, 1)

        return metrics, avg_loss

    def train(self):
        print("=" * 60)
        print("开始 DTI 训练")
        print("=" * 60)

        self._move_data_to_device()

        epochs = int(getattr(self.args, 'epochs', 150))
        best_metrics = None
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            train_metrics, train_loss = self._train_epoch(epoch)
            val_metrics, val_loss = self._evaluate(self.dataset.val_loader)
            test_metrics, test_loss = self._evaluate(self.dataset.test_loader)

            if epoch % 10 == 0 or epoch == 1:
                print(f"\nEpoch {epoch}/{epochs}")
                print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_metrics['Accuracy']:.4f}, "
                      f"F1: {train_metrics['F1']:.4f}, AUC: {train_metrics['AUC']:.4f}")
                print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_metrics['Accuracy']:.4f}, "
                      f"F1: {val_metrics['F1']:.4f}, AUC: {val_metrics['AUC']:.4f}")
                print(f"  Test  - Acc: {test_metrics['Accuracy']:.4f}, Prec: {test_metrics['Precision']:.4f}, "
                      f"Recall: {test_metrics['Recall']:.4f}, F1: {test_metrics['F1']:.4f}, "
                      f"AUC: {test_metrics['AUC']:.4f}, AP: {test_metrics['AP']:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                best_metrics = test_metrics
                best_epoch = epoch

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\n已加载最佳模型 (Epoch {best_epoch}, Val Loss: {self.best_val_loss:.4f})")

        final_metrics, _ = self._evaluate(self.dataset.test_loader)

        print("\n" + "=" * 60)
        print("DTI 训练完成 - 最终测试结果")
        print("=" * 60)
        print(f"  Accuracy:  {final_metrics['Accuracy']:.4f}")
        print(f"  Precision: {final_metrics['Precision']:.4f}")
        print(f"  Recall:    {final_metrics['Recall']:.4f}")
        print(f"  F1:        {final_metrics['F1']:.4f}")
        print(f"  AUC:       {final_metrics['AUC']:.4f}")
        print(f"  AP:        {final_metrics['AP']:.4f}")

        self._save_results(final_metrics, best_epoch)

        return final_metrics

    def _save_results(self, final_metrics, best_epoch):
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
            f.write(f"  model: {getattr(self.args, 'model', 'DTI')}\n")
            f.write(f"  task: {getattr(self.args, 'task', 'dti')}\n")
            f.write(f"  dataset: {getattr(self.args, 'matrix', 'dti')}\n")
            f.write(f"  modality: {getattr(self.args, 'modality', [])}\n")
            f.write(f"  noise_std: {getattr(self.args, 'noise_std', 0.0)}\n")
            f.write(f"  noise_ratio: {getattr(self.args, 'noise_ratio', 0.0)}\n")
            f.write(f"  noise_type: {getattr(self.args, 'noise_type', 'symmetric')}\n")
            f.write(f"  noise_edge: {getattr(self.args, 'noise_edge', 0.0)}\n")
            f.write(f"  sparse_drop_rate: {getattr(self.args, 'sparse_drop_rate', 0.0)}\n")
            f.write(f"  sparse_sample_rate: {getattr(self.args, 'sparse_sample_rate', 0.0)}\n\n")

            f.write(f"Results (classification):\n")
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
