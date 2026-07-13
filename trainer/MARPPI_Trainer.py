"""
MARPPI Trainer

支持:
- PPI 二分类: CrossEntropyLoss
- PPI 多标签分类: BCEWithLogitsLoss
- 评估指标: Accuracy, Precision, Recall, F1, AUC, AP (二分类) / AUC-ML, AP-ML (多标签)
"""

import copy
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)
from trainer.BaseTrainer import BaseTrainer


class MARPPI_Trainer(BaseTrainer):
    def __init__(self, args, logger, dataset, model, optimizer):
        super().__init__(args, logger, dataset, model, optimizer)

        self.task_type = getattr(self.args, 'task_type', 'binary')
        self.num_classes = getattr(self.args, 'num_classes', 2)

        if self.task_type == 'multilabel':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.best_model_state = None
        self.best_val_loss = float('inf')

    def _get_loss_function(self):
        return self.criterion

    def _prepare_batch_data(self, inp, task_type=None):
        labels = inp[2]
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        if self.task_type == 'multilabel':
            labels = labels.float()
        else:
            labels = labels.long()
        if self.device.type == 'cuda':
            labels = labels.cuda()
        return labels

    def _compute_metrics_binary(self, y_true, y_pred, y_prob=None):
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

    def _compute_metrics_multilabel(self, y_true, y_pred):
        metrics = {}
        try:
            metrics['AUC-ML'] = roc_auc_score(y_true, y_pred, average='macro', multi_class='ovr')
            metrics['AUC-ML-micro'] = roc_auc_score(y_true, y_pred, average='micro', multi_class='ovr')
        except ValueError:
            metrics['AUC-ML'] = 0.0
            metrics['AUC-ML-micro'] = 0.0

        try:
            metrics['AP-ML'] = average_precision_score(y_true, y_pred, average='macro')
        except ValueError:
            metrics['AP-ML'] = 0.0

        for avg in ['micro', 'macro']:
            p = precision_score(y_true, y_pred, average=avg, zero_division=0)
            r = recall_score(y_true, y_pred, average=avg, zero_division=0)
            f = f1_score(y_true, y_pred, average=avg, zero_division=0)
            metrics[f'Precision-{avg}'] = p
            metrics[f'Recall-{avg}'] = r
            metrics[f'F1-{avg}'] = f

        return metrics

    def _train_epoch(self, epoch, task_type=None):
        self.model.train()
        epoch_loss_sum = 0.0
        epoch_batches = 0
        ys, ypreds, yprobs = [], [], []

        for inp in self.dataset.train_loader:
            labels = self._prepare_batch_data(inp)

            self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(self.dataset.data_o, inp)
            loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()

            epoch_loss_sum += float(loss.item())
            epoch_batches += 1

            if self.task_type == 'multilabel':
                preds = (torch.sigmoid(logits) > 0.5).long().detach().cpu().numpy()
                probs = torch.sigmoid(logits).detach().cpu().numpy()
            else:
                preds = logits.argmax(dim=1).detach().cpu().numpy()
                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

            ypreds.append(preds)
            yprobs.append(probs)
            ys.append(labels.detach().cpu().numpy())

        y_true = np.concatenate(ys) if ys else np.array([])
        y_pred = np.concatenate(ypreds) if ypreds else np.array([])
        y_prob = np.concatenate(yprobs) if yprobs else np.array([])

        if self.task_type == 'multilabel':
            metrics = self._compute_metrics_multilabel(y_true, y_pred)
        else:
            metrics = self._compute_metrics_binary(y_true, y_pred, y_prob)

        avg_loss = epoch_loss_sum / max(epoch_batches, 1)
        return metrics, avg_loss

    def _evaluate(self, loader, task_type=None):
        self.model.eval()
        ys, ylogits, loss_sum, n = [], [], 0.0, 0

        with torch.no_grad():
            for inp in loader:
                labels = self._prepare_batch_data(inp)

                logits = self.model(self.dataset.data_o, inp)
                loss = self.criterion(logits, labels)

                loss_sum += float(loss.item())
                n += 1

                ylogits.append(logits.detach().cpu())
                ys.append(labels.detach().cpu().numpy())

        y_true = np.concatenate(ys) if ys else np.array([])
        y_logits = torch.cat(ylogits, dim=0) if ylogits else torch.tensor([])

        if self.task_type == 'multilabel':
            y_pred = (torch.sigmoid(y_logits) > 0.5).long().numpy()
            y_prob = torch.sigmoid(y_logits).numpy()
            metrics = self._compute_metrics_multilabel(y_true, y_pred)
        else:
            y_pred = y_logits.argmax(dim=1).numpy()
            y_prob = torch.softmax(y_logits, dim=1)[:, 1].numpy()
            metrics = self._compute_metrics_binary(y_true, y_pred, y_prob)

        avg_loss = loss_sum / max(n, 1)
        return metrics, avg_loss

    def train(self):
        task_desc = 'MARPPI 二分类' if self.task_type == 'binary' else 'MARPPI 多标签'
        print("=" * 60)
        print(f"开始 {task_desc} 训练")
        print("=" * 60)

        self._move_data_to_device()

        epochs = int(getattr(self.args, 'epochs', 100))
        best_metrics = None
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            train_metrics, train_loss = self._train_epoch(epoch)
            val_metrics, val_loss = self._evaluate(self.dataset.val_loader)
            test_metrics, test_loss = self._evaluate(self.dataset.test_loader)

            if epoch % 10 == 0 or epoch == 1:
                print(f"\nEpoch {epoch}/{epochs}")
                print(f"  Train - Loss: {train_loss:.4f}", end='')
                if self.task_type == 'binary':
                    print(f", Acc: {train_metrics['Accuracy']:.4f}, "
                          f"F1: {train_metrics['F1']:.4f}, AUC: {train_metrics['AUC']:.4f}")
                else:
                    print(f", F1-micro: {train_metrics['F1-micro']:.4f}, "
                          f"F1-macro: {train_metrics['F1-macro']:.4f}, "
                          f"AUC-ML: {train_metrics['AUC-ML']:.4f}")
                print(f"  Val   - Loss: {val_loss:.4f}")
                print(f"  Test  - " + self._format_metrics(test_metrics))

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
        print(f"{task_desc}训练完成 - 最终测试结果")
        print("=" * 60)
        print(self._format_metrics(final_metrics))

        self._save_results(final_metrics, best_epoch)
        return final_metrics

    def _format_metrics(self, metrics):
        lines = []
        for k, v in metrics.items():
            if isinstance(v, float):
                lines.append(f"{k}: {v:.4f}")
            else:
                lines.append(f"{k}: {v}")
        return ', '.join(lines)

    def _save_results(self, final_metrics, best_epoch):
        out_file = getattr(self.args, 'out_file', 'results.txt')
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        elapsed_time = self._get_elapsed_time()
        memory_usage = self._get_memory_usage()

        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open(out_file, 'a') as f:
            f.write("=" * 70 + "\n")
            f.write(f"Run at: {timestamp}\n")
            f.write("=" * 70 + "\n")
            f.write(f"Config Info:\n")
            f.write(f"  model: {getattr(self.args, 'model', 'MARPPI')}\n")
            f.write(f"  task: {getattr(self.args, 'task', 'ppi')}\n")
            f.write(f"  dataset: {getattr(self.args, 'matrix', 'ppi')}\n")
            f.write(f"  ppi_type: {getattr(self.args, 'ppi_type', 'binary')}\n")
            f.write(f"  modality: {getattr(self.args, 'modality', [])}\n")
            f.write(f"  num_classes: {self.num_classes}\n")
            f.write(f"  noise_std: {getattr(self.args, 'noise_std', 0.0)}\n")
            f.write(f"  noise_ratio: {getattr(self.args, 'noise_ratio', 0.0)}\n")
            f.write(f"  noise_type: {getattr(self.args, 'noise_type', 'symmetric')}\n")
            f.write(f"  noise_edge: {getattr(self.args, 'noise_edge', 0.0)}\n")
            f.write(f"  sparse_drop_rate: {getattr(self.args, 'sparse_drop_rate', 0.0)}\n")
            f.write(f"  sparse_sample_rate: {getattr(self.args, 'sparse_sample_rate', 0.0)}\n\n")

            f.write(f"Results:\n")
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
