#负责判断具体需要哪个dataset
import argparse
from data.MRCGNN_dataset import MRCGNN_dataset
from data.ZeroDDI_dataset import ZeroDDI_dataset
from data.Unified_dataset import Unified_dataset
from data.TIGER_dataset import TIGER_dataset
from data.GoGNN_dataset import GoGNN_dataset
from data.MUFFIN_dataset import MUFFIN_dataset
from data.MVA_dataset import MVA_dataset
from data.DTA_dataset import DTADataset
from data.DTI_dataset import DTIDataset
from data.ColdstartCPI_dataset import ColdstartCPI_Dataset
from data.PPI_dataset import PPIDataset
from data.DL_PPI_dataset import DL_PPIDataset
from data.TAGPPI_dataset import TAGPPI_Dataset
class dataset_manager:
    def __init__(self,
                 args: argparse.ArgumentParser):
        self.dataset = None
        self.args = args
        self.dataset_mapping = {"MRCGNN": MRCGNN_dataset,
                                "GOGNN": Unified_dataset,
                                "ZeroDDI": ZeroDDI_dataset,
                                "DDIMDL": Unified_dataset,
                                "TIGER": Unified_dataset, 
                                "ConvLSTM": Unified_dataset,    
                                "MVA": Unified_dataset,
                                "MUFFIN": Unified_dataset,
                                "DeepDDI": Unified_dataset,
                                "DDKG": Unified_dataset,
                                "SumGNN": Unified_dataset,
                                "KGNN": Unified_dataset,
                                "LaGAT": Unified_dataset,
                                "PHGLDDI": Unified_dataset,
                                "MMDGDTI": Unified_dataset,
                                "DSNDDI": Unified_dataset,
                                "ExDDI": Unified_dataset,
                                "MIRACLE": Unified_dataset,
                                "CASTER": Unified_dataset,
                                "MKGFENN": Unified_dataset,
                                "DTA": DTADataset,
                                "DTI": DTIDataset,
                                "ColdstartCPI": ColdstartCPI_Dataset,
                                "PPI": PPIDataset,
                                "DL_PPI": DL_PPIDataset,
                                "TAGPPI": TAGPPI_Dataset,
                                "PPI_TUnA": PPIDataset,
                                "MARPPI": PPIDataset,
                                "MAPE_PPI": PPIDataset,
                                "HIGH_PPI": PPIDataset,
                                "GTB_PPI": PPIDataset,
                                "GraphPPIS": PPIDataset,
                                "D_SCRIPT": PPIDataset,
                                "CollaPPI": PPIDataset,
                                "AdaMBind": Unified_dataset,
                                }

    def load_dataset(self):
        # DL-PPI 模型使用专门的 DL_PPIDataset
        if self.args.model == "DL_PPI":
            print(f"[Dataset Manager] 加载 DL-PPI 数据集")
            return DL_PPIDataset(self.args)
        # TAGPPI 模型使用专门的 TAGPPI_Dataset
        if self.args.model == "TAGPPI":
            print(f"[Dataset Manager] 加载 TAGPPI 数据集")
            return TAGPPI_Dataset(self.args)
        # PPI_TUnA 模型复用 PPIDataset（相同的嵌入和边列表格式）
        if self.args.model == "PPI_TUnA":
            print(f"[Dataset Manager] 加载 PPI 数据集 (供 PPI_TUnA 使用)")
            return PPIDataset(self.args)
        # MARPPI 模型复用 PPIDataset（相同的嵌入和边列表格式）
        if self.args.model == "MARPPI":
            print(f"[Dataset Manager] 加载 PPI 数据集 (供 MARPPI 使用)")
            return PPIDataset(self.args)
        # MAPE_PPI 模型复用 PPIDataset（相同的嵌入和边列表格式）
        if self.args.model == "MAPE_PPI":
            print(f"[Dataset Manager] 加载 PPI 数据集 (供 MAPE_PPI 使用)")
            return PPIDataset(self.args)
        # HIGH_PPI 模型复用 PPIDataset（相同的嵌入和边列表格式）
        if self.args.model == "HIGH_PPI":
            print(f"[Dataset Manager] 加载 PPI 数据集 (供 HIGH_PPI 使用)")
            return PPIDataset(self.args)
        # GTB_PPI 模型复用 PPIDataset（相同的嵌入和边列表格式）
        if self.args.model == "GTB_PPI":
            print(f"[Dataset Manager] 加载 PPI 数据集 (供 GTB_PPI 使用)")
            return PPIDataset(self.args)
        # GraphPPIS 模型复用 PPIDataset（相同的嵌入和边列表格式）
        if self.args.model == "GraphPPIS":
            print(f"[Dataset Manager] 加载 PPI 数据集 (供 GraphPPIS 使用)")
            return PPIDataset(self.args)
        # D_SCRIPT 模型复用 PPIDataset（相同的嵌入和边列表格式）
        if self.args.model == "D_SCRIPT":
            print(f"[Dataset Manager] 加载 PPI 数据集 (供 D_SCRIPT 使用)")
            return PPIDataset(self.args)
        # CollaPPI 模型复用 PPIDataset（相同的嵌入和边列表格式）
        if self.args.model == "CollaPPI":
            print(f"[Dataset Manager] 加载 PPI 数据集 (供 CollaPPI 使用)")
            return PPIDataset(self.args)
        # ColdstartCPI 模型使用专门的 ColdstartCPI_Dataset（优先判断）
        if self.args.model == "ColdstartCPI":
            print(f"[Dataset Manager] 加载 ColdstartCPI 数据集")
            return ColdstartCPI_Dataset(self.args)
        # PPI 任务使用专门的 PPIDataset
        if getattr(self.args, 'task', 'train_xxxx') in ('ppi_b', 'ppi_m'):
            print(f"[Dataset Manager] 加载 PPI 数据集")
            return PPIDataset(self.args)
        # 代谢和神经退行性疾病 PPI 数据集 - 自动设置为 ppi_b 任务
        if getattr(self.args, 'matrix', None) in ('metabolism', 'neurodegenerative'):
            print(f"[Dataset Manager] 检测到 PPI 专项数据集: {self.args.matrix}，自动设置为 ppi_b 任务")
            self.args.task = 'ppi_b'
            self.args.ppi_type = 'binary'
            return PPIDataset(self.args)
        # SHS148k PPI 多标签数据集 - 自动设置为 ppi_m 任务
        if getattr(self.args, 'matrix', None) == 'SHS148k':
            print(f"[Dataset Manager] 检测到 PPI 多标签数据集: {self.args.matrix}，自动设置为 ppi_m 任务")
            self.args.task = 'ppi_m'
            self.args.ppi_type = 'multilabel'
            return PPIDataset(self.args)
        # BindingDB 和 BIOSNAP DTI 数据集 - 自动设置为 dti 任务
        if getattr(self.args, 'matrix', None) in ('bindingdb', 'BIOSNAP'):
            print(f"[Dataset Manager] 检测到 DTI 专项数据集: {self.args.matrix}，自动设置为 dti 任务")
            self.args.task = 'dti'
            return DTIDataset(self.args)
        # DTA 任务使用专门的 DTA 数据集
        if getattr(self.args, 'task', 'train_xxxx') == 'dta':
            print(f"[Dataset Manager] 加载 DTA 数据集")
            return DTADataset(self.args)
        # DTI 任务使用专门的 DTI 数据集
        if getattr(self.args, 'task', 'train_xxxx') == 'dti':
            print(f"[Dataset Manager] 加载 DTI 数据集")
            return DTIDataset(self.args)
        # AdaMBind 模型支持 DTI 二分类 和 DTA 回归
        if self.args.model == "AdaMBind":
            task = getattr(self.args, 'task', 'dta')
            if task == 'dti':
                print(f"[Dataset Manager] 加载 DTI 数据集 (供 AdaMBind 使用)")
                return DTIDataset(self.args)
            else:
                print(f"[Dataset Manager] 加载 DTA 数据集 (供 AdaMBind 使用)")
                return DTADataset(self.args)
        if self.args.model in ["TIGER"] and self.args.origin:
            return TIGER_dataset(self.args)
        if self.args.model in ["GOGNN"] and self.args.origin:
            return GoGNN_dataset(self.args)
        if self.args.model in ["MUFFIN"] and self.args.origin:
            return MUFFIN_dataset(self.args)
        if self.args.model in ["MVA"] and self.args.origin:
            return MVA_dataset(self.args)
        self.dataset = self.dataset_mapping[self.args.model](self.args)
        return self.dataset