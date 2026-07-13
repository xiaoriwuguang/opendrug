"""
DL-PPI 训练器

适配 DL-PPI 模型:
- 模型 forward: model(x, edge_index, edge_batch_idx, dropout)
- 图结构: 使用 data_o 中的 edge_index 和 edge_attr_1
- 边索引: 训练/验证/测试集返回的是在 edge_index 中的真实索引

支持 PPI 二分类（CrossEntropyLoss）和多标签分类（BCEWithLogitsLoss）。
"""

import copy
import os
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)
from trainer.BaseTrainer import BaseTrainer


class DL_PPI_Trainer(BaseTrainer):
    def __init__(self, args, logger, dataset, model, optimizer):
        super().__init__(args, logger, dataset, model, optimizer)

        self.task_type = getattr(self.args, 'task_type', 'binary')
        self.num_classes = getattr(self.args, 'num_classes', 2)

        # 计算类别权重（处理类别不平衡）
        self._setup_loss()

        self.best_model_state = None
        self.best_val_metric = -1   # 用 AUC/F1 早停，不再用 loss

    def _setup_loss(self, weight=None):
        if self.task_type == 'multilabel':
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=weight)

    def _compute_class_weights(self):
        """计算类别权重，处理类别不平衡（用于 CrossEntropyLoss）"""
        edge_labels = self.dataset.data_o.edge_attr_1
        if self.task_type == 'multilabel':
            return None
        if edge_labels.dim() == 1:
            labels = edge_labels.cpu().numpy()
        else:
            labels = edge_labels[:, 0].cpu().numpy()
        total = len(labels)
        n_neg = int((labels == 0).sum())
        n_pos = int((labels == 1).sum())
        if n_pos == 0 or n_neg == 0:
            return None
        weight = total / (2 * self.num_classes)
        w_neg = weight / max(n_neg, 1)
        w_pos = weight / max(n_pos, 1)
        class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float32, device=self.device)
        print(f"[DL_PPI] class_weights=class0:{w_neg:.3f}, class1:{w_pos:.3f} (neg={n_neg}, pos={n_pos})")
        return class_weights

    def _prepare_batch_data(self, inp):
        p1_idx, p2_idx, labels = inp

        if isinstance(p1_idx, np.ndarray):
            p1_idx = torch.from_numpy(p1_idx)
        if isinstance(p2_idx, np.ndarray):
            p2_idx = torch.from_numpy(p2_idx)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        labels = labels.float()
        if self.device.type == 'cuda':
            p1_idx = p1_idx.cuda()
            p2_idx = p2_idx.cuda()
            labels = labels.cuda()

        return p1_idx, p2_idx, labels

    def _compute_metrics(self, y_true, y_pred, y_prob=None):
        """同时支持二分类和多标签指标"""
        if self.task_type == 'multilabel':
            return self._compute_metrics_multilabel(y_true, y_pred, y_prob)
        else:
            return self._compute_metrics_binary(y_true, y_pred, y_prob)

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

    def _compute_metrics_multilabel(self, y_true, y_pred, y_prob=None):
        metrics = {}
        n_classes = y_true.shape[1]

        for avg in ['micro', 'macro']:
            p = precision_score(y_true, y_pred, average=avg, zero_division=0)
            r = recall_score(y_true, y_pred, average=avg, zero_division=0)
            f = f1_score(y_true, y_pred, average=avg, zero_division=0)
            metrics[f'Precision-{avg}'] = p
            metrics[f'Recall-{avg}'] = r
            metrics[f'F1-{avg}'] = f

        if y_prob is not None:
            try:
                metrics['AUC-ML'] = roc_auc_score(y_true, y_prob, average='macro')
                metrics['AUC-ML-micro'] = roc_auc_score(y_true, y_prob, average='micro')
            except ValueError:
                metrics['AUC-ML'] = 0.0
                metrics['AUC-ML-micro'] = 0.0
            try:
                metrics['AP-ML'] = average_precision_score(y_true, y_prob, average='macro')
            except ValueError:
                metrics['AP-ML'] = 0.0
        return metrics

    def _train_epoch(self, epoch):
        self.model.train()
        epoch_loss_sum = 0.0
        epoch_batches = 0
        ys, ypreds, yprobs = [], [], []

        train_mask = self.dataset.data_o.train_mask

        if len(train_mask) == 0:
            return {}, 0.0

        random.shuffle(train_mask)
        steps = int(np.ceil(len(train_mask) / self.args.batch))

        for step in range(steps):
            if step == steps - 1:
                edge_batch_idx = train_mask[step * self.args.batch:]
            else:
                edge_batch_idx = train_mask[step * self.args.batch: step * self.args.batch + self.args.batch]

            edge_batch_idx_t = torch.tensor(edge_batch_idx, dtype=torch.long,
                                           device=self.device)

            output = self.model(
                self.dataset.data_o.x,
                self.dataset.data_o.edge_index,
                edge_batch_idx_t,
                dropout=float(getattr(self.args, 'dropout', 0.5))
            )

            if self.task_type == 'binary':
                label = self.dataset.data_o.edge_attr_1[edge_batch_idx].long().to(self.device)
            else:
                label = self.dataset.data_o.edge_attr_1[edge_batch_idx].float().to(self.device)

            loss = self.loss_fn(output, label)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            epoch_loss_sum += float(loss.item())
            epoch_batches += 1

            if self.task_type == 'multilabel':
                probs = torch.sigmoid(output).detach().cpu().numpy()
                preds = (probs > 0.5).astype(int)
                ypreds.append(preds)
                yprobs.append(probs)
                ys.append(label.detach().cpu().numpy())
            else:
                probs = torch.softmax(output, dim=1)[:, 1].detach().cpu().numpy().flatten()
                preds = (probs > 0.5).astype(int)
                ypreds.append(preds)
                yprobs.append(probs)
                ys.append(label.detach().cpu().numpy().flatten())

        y_true = np.concatenate(ys) if ys else np.array([])
        y_pred = np.concatenate(ypreds) if ypreds else np.array([])
        y_prob = np.concatenate(yprobs) if yprobs else np.array([])

        metrics = self._compute_metrics(y_true, y_pred, y_prob)
        avg_loss = epoch_loss_sum / max(epoch_batches, 1)
        return metrics, avg_loss

    def _evaluate(self, split='val'):
        self.model.eval()
        ys, outputs_all, loss_sum, n = [], [], 0.0, 0

        if split == 'val':
            mask = self.dataset.data_o.val_mask
        else:
            mask = self.dataset.data_o.test_mask

        if len(mask) == 0:
            return {}, 0.0

        steps = int(np.ceil(len(mask) / self.args.batch))

        with torch.no_grad():
            for step in range(steps):
                if step == steps - 1:
                    edge_batch_idx = mask[step * self.args.batch:]
                else:
                    edge_batch_idx = mask[step * self.args.batch: step * self.args.batch + self.args.batch]

                edge_batch_idx_t = torch.tensor(edge_batch_idx, dtype=torch.long,
                                               device=self.device)

                output = self.model(
                    self.dataset.data_o.x,
                    self.dataset.data_o.edge_index,
                    edge_batch_idx_t,
                    dropout=0.0
                )

                if self.task_type == 'binary':
                    label = self.dataset.data_o.edge_attr_1[edge_batch_idx].long().to(self.device)
                else:
                    label = self.dataset.data_o.edge_attr_1[edge_batch_idx].float().to(self.device)

                loss = self.loss_fn(output, label)
                loss_sum += float(loss.item())
                n += 1

                outputs_all.append(output.detach().cpu())
                ys.append(label.detach().cpu().numpy())

        y_logits = torch.cat(outputs_all, dim=0)
        y_true = np.concatenate(ys) if ys else np.array([])

        if self.task_type == 'multilabel':
            y_prob = torch.sigmoid(y_logits).numpy()
            y_pred = (y_prob > 0.5).astype(int)
        else:
            y_prob = torch.softmax(y_logits, dim=1)[:, 1].numpy().flatten()
            y_pred = (y_prob > 0.5).astype(int)

        metrics = self._compute_metrics(y_true, y_pred, y_prob)
        avg_loss = loss_sum / max(n, 1)
        return metrics, avg_loss

    def train(self):
        task_desc = 'DL-PPI 二分类' if self.task_type == 'binary' else 'DL-PPI 多标签'
        print("=" * 60)
        print(f"开始 {task_desc} 训练")
        print("=" * 60)

        self._move_data_to_device()

        # 计算 class_weights 并初始化损失函数
        if self.task_type == 'multilabel':
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        epochs = int(getattr(self.args, 'epochs', 150))
        best_metrics = None
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            train_metrics, train_loss = self._train_epoch(epoch)
            val_metrics, val_loss = self._evaluate('val')
            test_metrics, test_loss = self._evaluate('test')

            if epoch % 10 == 0 or epoch == 1:
                print(f"\nEpoch {epoch}/{epochs}")
                print(f"  Train - Loss: {train_loss:.4f}", end='')
                if self.task_type == 'binary':
                    print(f", Acc: {train_metrics.get('Accuracy', 0):.4f}, "
                          f"F1: {train_metrics.get('F1', 0):.4f}")
                else:
                    print(f", F1-micro: {train_metrics.get('F1-micro', 0):.4f}, "
                          f"F1-macro: {train_metrics.get('F1-macro', 0):.4f}")
                print(f"  Val   - Loss: {val_loss:.4f}")
                print(f"  Test  - {self._format_metrics(test_metrics)}")

            # 早停策略：用 AUC 而非 loss（解决类别不平衡时 loss 欺骗问题）
            if self.task_type == 'multilabel':
                val_auc = val_metrics.get('AUC-ML-micro', 0.0)
            else:
                val_auc = val_metrics.get('AUC', 0.0)
            if val_auc > self.best_val_metric:
                self.best_val_metric = val_auc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                best_metrics = test_metrics
                best_epoch = epoch

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            auc_key = 'AUC-ML-micro' if self.task_type == 'multilabel' else 'AUC'
            print(f"\n已加载最佳模型 (Epoch {best_epoch}, Val {auc_key}: {self.best_val_metric:.4f})")

        final_metrics, _ = self._evaluate('test')

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
            f.write(f"  model: DL-PPI\n")
            f.write(f"  task: {getattr(self.args, 'task', 'ppi_b')}\n")
            f.write(f"  ppi_type: {getattr(self.args, 'ppi_type', 'binary')}\n")
            f.write(f"  dataset: {getattr(self.args, 'matrix', 'unknown')}\n")
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
