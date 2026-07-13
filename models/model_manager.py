import argparse
import torch
from models.gognn.gognn import GoGNN as origognn
from models.muffin.muffin import MUFFIN as orimuffin
from models.mva.mva import MVA as orimva
from models.MRCGNN import MRCGNN
from models.GOGNN import GOGNN
from models.ZeroDDI import ZeroDDI
from models.DDIMDL import DDIMDL    
from models.TIGER import TIGER
from models.tiger.tiger import TIGER as oritiger
from models.ConvLSTM import ConvLSTM
from models.MVA import MVA
from models.MUFFIN import MUFFIN
from models.DeepDDI import DeepDDI
from models.DDKG import DDKG
from models.SumGNN import SumGNN
from models.KGNN import KGNN
from models.DSNDDI import DSNDDI
from models.LaGAT import LaGAT
from models.PHGLDDI import PHGLDDI
from models.MMDGDTI import MMDGDTI 
from models.ExDDI import ExDDI
from models.MIRACLE import MIRACLE
from models.CASTER import CASTER 
from models.MKGFENN import MKGFENN
from models.DTA import DTABilinearPredictor
from models.DTI import DTIPredictor
from models.KGE_NFM import KGE_NFM, KGE_NFM_Model
from models.MGraphDTA import MGraphDTA, MGraphDTA_Model
from models.MMD_DTA import MMD_DTA, MMD_DTA_Model
from models.RSGCL_DTI import RSGCL_DTI, RSGCL_DTI_Model
from models.GraphDTA import GraphDTA, GraphDTA_Model
from models.EviDTI import EviDTI, EviDTI_Model
from models.DTIAM import DTIAM, DTIAM_Model
from models.DrugBAN import DrugBAN, DrugBAN_Model, DrugBAN_Embedding_Model
from models.ColdstartCPI import ColdstartCPI, ColdstartCPI_Model
from models.PPI_model import PPIPredictor, PPI_Model
from models.DL_PPI import GIN_Net2, DL_PPI_Model
from models.TAGPPI import TAGPPI, TAGPPI_Model
from models.PPI_TUnA import PPITUnA, PPI_TUnA_Model
from models.MARPPI import MARPPI, MARPPI_Model
from models.MAPE_PPI import MAPE_PPI, MAPE_PPI_Model
from models.HIGH_PPI import HIGH_PPI, HIGH_PPI_Model
from models.GTB_PPI import GTB_PPI, GTB_PPI_Model
from models.GraphPPIS import GraphPPIS, GraphPPIS_Model
from models.D_SCRIPT import D_SCRIPT, D_SCRIPT_Model
from models.CollaPPI import CollaPPI, CollaPPI_Model
from models.AdaMBind import AdaMBind, AdaMBind_Model
from inspect import signature

class model_manager:
    def __init__(self,
                 args:argparse):
        self.args = args    
        self.model_mapping = {"MRCGNN": MRCGNN,
                              "GOGNN" : GOGNN,
                              "ZeroDDI": ZeroDDI,
                              "DDIMDL": DDIMDL,
                              "TIGER": TIGER,
                              "ConvLSTM": ConvLSTM,
                              "MVA": MVA,
                              "MUFFIN": MUFFIN,
                              "DeepDDI": DeepDDI,
                              "DDKG": DDKG,
                              "SumGNN": SumGNN,
                              "KGNN": KGNN,
                              "LaGAT": LaGAT,
                              "PHGLDDI": PHGLDDI,
                              "MMDGDTI": MMDGDTI,
                              "DSNDDI": DSNDDI,
                              "ExDDI": ExDDI,
                              "MIRACLE": MIRACLE,
                              "CASTER": CASTER,
                              "MKGFENN": MKGFENN,
                              "DTA": DTABilinearPredictor,
                              "DTI": DTIPredictor,
                              "KGE_NFM": KGE_NFM_Model,
                              "MGraphDTA": MGraphDTA_Model,
                              "MMD_DTA": MMD_DTA_Model,
                              "RSGCL_DTI": RSGCL_DTI_Model,
                              "GraphDTA": GraphDTA_Model,
                              "EviDTI": EviDTI_Model,
                              "DTIAM": DTIAM_Model,
                              "DrugBAN": DrugBAN_Model,
                              "ColdstartCPI": ColdstartCPI_Model,
                              "PPI": PPI_Model,
                              "DL_PPI": DL_PPI_Model,
                              "TAGPPI": TAGPPI_Model,
                              "PPI_TUnA": PPI_TUnA_Model,
                              "MARPPI": MARPPI_Model,
                              "MAPE_PPI": MAPE_PPI_Model,
                              "HIGH_PPI": HIGH_PPI_Model,
                              "GTB_PPI": GTB_PPI_Model,
                              "GraphPPIS": GraphPPIS_Model,
                              "D_SCRIPT": D_SCRIPT_Model,
                              "CollaPPI": CollaPPI_Model,
                              "AdaMBind": AdaMBind_Model,
                              }

    def load_model(self):
        # 先处理特殊任务（DTA/KGE_NFM回归任务不需要num_classes）
        # --- DTA 任务特殊处理 ---
        if self.args.model == "DTA":
            drug_dim = int(getattr(self.args, 'drug_dim', 512))
            protein_dim = int(getattr(self.args, 'protein_dim', 512))
            kwargs = {
                'drug_dim': drug_dim,
                'protein_dim': protein_dim,
                'hidden_dim': int(getattr(self.args, 'hidden1', 256)),
                'dropout': float(getattr(self.args, 'dropout', 0.3)),
            }
            cls = self.model_mapping[self.args.model]
            model = cls(**kwargs)
            device = getattr(self.args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            return model

        # --- KGE_NFM 任务特殊处理 ---
        if self.args.model == "KGE_NFM":
            drug_dim = int(getattr(self.args, 'drug_dim', 512))
            protein_dim = int(getattr(self.args, 'protein_dim', 512))
            task_type = getattr(self.args, 'task', 'train_xxxx')

            kwargs = {
                'drug_dim': drug_dim,
                'protein_dim': protein_dim,
                'hidden_dim': int(getattr(self.args, 'hidden1', 256)),
                'nfm_hidden': int(getattr(self.args, 'hidden2', 128)),
                'dnn_layers': int(getattr(self.args, 'layers', 2)),
                'dropout': float(getattr(self.args, 'dropout', 0.3)),
                'model_variant': getattr(self.args, 'kge_nfm_variant', 'standard'),
            }

            if task_type == 'dta':
                kwargs['task_type'] = 'regression'
                kwargs['num_classes'] = 1
            else:
                kwargs['task_type'] = 'classification'
                kwargs['num_classes'] = 2

            cls = self.model_mapping[self.args.model]
            model = cls(**kwargs)
            device = getattr(self.args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            return model

        # --- MGraphDTA 任务特殊处理 ---
        if self.args.model == "MGraphDTA":
            drug_dim = int(getattr(self.args, 'drug_dim', 512))
            protein_dim = int(getattr(self.args, 'protein_dim', 512))
            task_type = getattr(self.args, 'task', 'train_xxxx')

            kwargs = {
                'drug_dim': drug_dim,
                'protein_dim': protein_dim,
                'hidden_dim': int(getattr(self.args, 'hidden1', 256)),
                'dropout': float(getattr(self.args, 'dropout', 0.1)),
                'model_variant': getattr(self.args, 'mgraphdta_variant', 'embedding'),
            }

            if task_type == 'dta':
                kwargs['task_type'] = 'regression'
                kwargs['num_classes'] = 1
            else:
                kwargs['task_type'] = 'classification'
                kwargs['num_classes'] = 2

            cls = self.model_mapping[self.args.model]
            model = cls(**kwargs)
            device = getattr(self.args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            return model

        # --- MMD_DTA 任务特殊处理 ---
        if self.args.model == "MMD_DTA":
            drug_dim = int(getattr(self.args, 'drug_dim', 512))
            protein_dim = int(getattr(self.args, 'protein_dim', 512))
            task_type = getattr(self.args, 'task', 'train_xxxx')

            kwargs = {
                'drug_dim': drug_dim,
                'protein_dim': protein_dim,
                'hidden_dim': int(getattr(self.args, 'hidden1', 256)),
                'dropout': float(getattr(self.args, 'dropout', 0.2)),
            }

            if task_type == 'dta':
                kwargs['task_type'] = 'regression'
                kwargs['num_classes'] = 1
            else:
                kwargs['task_type'] = 'classification'
                kwargs['num_classes'] = 2

            cls = self.model_mapping[self.args.model]
            model = cls(**kwargs)
            device = getattr(self.args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            return model

        # --- RSGCL_DTI 任务特殊处理 ---
        if self.args.model == "RSGCL_DTI":
            drug_dim = int(getattr(self.args, 'drug_dim', 512))
            protein_dim = int(getattr(self.args, 'protein_dim', 512))
            task_type = getattr(self.args, 'task', 'train_xxxx')

            kwargs = {
                'drug_dim': drug_dim,
                'protein_dim': protein_dim,
                'hidden_dim': int(getattr(self.args, 'hidden1', 512)),
                'dropout': float(getattr(self.args, 'dropout', 0.2)),
            }

            if task_type == 'dta':
                kwargs['task_type'] = 'regression'
                kwargs['num_classes'] = 1
            else:
                kwargs['task_type'] = 'classification'
                kwargs['num_classes'] = 2

            cls = self.model_mapping[self.args.model]
            model = cls(**kwargs)
            device = getattr(self.args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            return model

        # --- GraphDTA 任务特殊处理 ---
        if self.args.model == "GraphDTA":
            drug_dim = int(getattr(self.args, 'drug_dim', 512))
            protein_dim = int(getattr(self.args, 'protein_dim', 512))
            task_type = getattr(self.args, 'task', 'train_xxxx')

            kwargs = {
                'drug_dim': drug_dim,
                'protein_dim': protein_dim,
                'hidden_dim': int(getattr(self.args, 'hidden1', 256)),
                'dropout': float(getattr(self.args, 'dropout', 0.2)),
            }

            if task_type == 'dta':
                kwargs['task_type'] = 'regression'
                kwargs['num_classes'] = 1
            else:
                kwargs['task_type'] = 'classification'
                kwargs['num_classes'] = 2

            cls = self.model_mapping[self.args.model]
            model = cls(**kwargs)
            device = getattr(self.args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            return model

        # --- EviDTI 任务特殊处理 ---
        if self.args.model == "EviDTI":
            drug_dim = int(getattr(self.args, 'drug_dim', 512))
            protein_dim = int(getattr(self.args, 'protein_dim', 512))
            task_type = getattr(self.args, 'task', 'train_xxxx')

            kwargs = {
                'drug_dim': drug_dim,
                'protein_dim': protein_dim,
                'hidden_dim': int(getattr(self.args, 'hidden1', 256)),
                'dropout': float(getattr(self.args, 'dropout', 0.2)),
            }

            if task_type == 'dta':
                kwargs['task_type'] = 'regression'
                kwargs['num_classes'] = 1
            else:
                kwargs['task_type'] = 'classification'
                kwargs['num_classes'] = 2

            cls = self.model_mapping[self.args.model]
            model = cls(**kwargs)
            device = getattr(self.args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            return model

        # --- DTIAM 任务特殊处理 ---
        if self.args.model == "DTIAM":
            drug_dim = int(getattr(self.args, 'drug_dim', 512))
            protein_dim = int(getattr(self.args, 'protein_dim', 512))
            task_type = getattr(self.args, 'task', 'train_xxxx')

            kwargs = {
                'drug_dim': drug_dim,
                'protein_dim': protein_dim,
                'hidden_dim': int(getattr(self.args, 'hidden1', 256)),
                'dropout': float(getattr(self.args, 'dropout', 0.2)),
            }

            if task_type == 'dta':
                kwargs['task_type'] = 'regression'
                kwargs['num_classes'] = 1
            else:
                kwargs['task_type'] = 'classification'
                kwargs['num_classes'] = 2

            cls = self.model_mapping[self.args.model]
            model = cls(**kwargs)
            device = getattr(self.args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            return model

        # --- DrugBAN 任务特殊处理 ---
        if self.args.model == "DrugBAN":
            drug_dim = int(getattr(self.args, 'drug_dim', 512))
            protein_dim = int(getattr(self.args, 'protein_dim', 512))
            task_type = getattr(self.args, 'task', 'train_xxxx')

            kwargs = {
                'drug_dim': drug_dim,
                'protein_dim': protein_dim,
                'hidden_dim': int(getattr(self.args, 'hidden1', 256)),
                'ban_heads': int(getattr(self.args, 'ban_heads', 2)),
                'ban_variant': getattr(self.args, 'ban_variant', 'standard'),
                'dropout': float(getattr(self.args, 'dropout', 0.2)),
            }

            if task_type == 'dta':
                kwargs['task_type'] = 'regression'
                kwargs['num_classes'] = 1
            else:
                kwargs['task_type'] = 'classification'
                kwargs['num_classes'] = 2

            cls = self.model_mapping[self.args.model]
            model = cls(**kwargs)
            device = getattr(self.args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            return model

        # --- ColdstartCPI 任务特殊处理 ---
        if self.args.model == "ColdstartCPI":
            drug_g_dim = int(getattr(self.args, 'drug_g_dim', 512))
            drug_m_dim = int(getattr(self.args, 'drug_m_dim', 512))
            protein_g_dim = int(getattr(self.args, 'protein_g_dim', 512))
            protein_m_dim = int(getattr(self.args, 'protein_m_dim', 512))
            unify_num = int(getattr(self.args, 'unify_num', 256))
            task_type = getattr(self.args, 'task', 'train_xxxx')

            kwargs = {
                'drug_g_dim': drug_g_dim,
                'drug_m_dim': drug_m_dim,
                'protein_g_dim': protein_g_dim,
                'protein_m_dim': protein_m_dim,
                'unify_num': unify_num,
                'head_num': int(getattr(self.args, 'ban_heads', 4)),
                'dropout': float(getattr(self.args, 'dropout', 0.1)),
                'max_drug_seq': int(getattr(self.args, 'max_drug_seq', 100)),
                'max_protein_seq': int(getattr(self.args, 'max_protein_seq', 1000)),
            }

            if task_type == 'dta':
                kwargs['task_type'] = 'regression'
                kwargs['num_classes'] = 1
            else:
                kwargs['task_type'] = 'classification'
                kwargs['num_classes'] = 2

            cls = self.model_mapping[self.args.model]
            model = cls(**kwargs)
            device = getattr(self.args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            return model

        # --- PPI 任务特殊处理 ---
        if self.args.model == "PPI":
            protein_dim = int(getattr(self.args, 'protein_dim', 512))
            task_type = getattr(self.args, 'task_type', 'binary')
            num_classes = int(getattr(self.args, 'num_classes', 2))

            kwargs = {
                'protein_dim': protein_dim,
                'hidden_dim': int(getattr(self.args, 'hidden1', 256)),
                'dropout': float(getattr(self.args, 'dropout', 0.3)),
                'num_classes': num_classes,
                'task_type': task_type,
            }

            cls = self.model_mapping[self.args.model]
            model = cls(**kwargs)
            device = getattr(self.args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            return model

        # --- DL_PPI 任务特殊处理 ---
        if self.args.model == "DL_PPI":
            protein_dim = int(getattr(self.args, 'protein_dim', 640))
            gin_in_feature = int(getattr(self.args, 'gin_in_feature', 256))
            hidden = int(getattr(self.args, 'hidden1', 512))
            num_layers = int(getattr(self.args, 'gin_layers', 1))
            pool_size = int(getattr(self.args, 'gin_pool_size', 3))
            num_classes = int(getattr(self.args, 'num_classes', 7))
            feature_fusion = getattr(self.args, 'dlppi_fusion', 'NTN')
            dropout = float(getattr(self.args, 'dropout', 0.5))

            kwargs = {
                'in_feature': 1,
                'gin_in_feature': gin_in_feature,
                'hidden': hidden,
                'num_layers': num_layers,
                'pool_size': pool_size,
                'num_classes': num_classes,
                'feature_fusion': feature_fusion,
                'dropout': dropout,
            }

            cls = self.model_mapping[self.args.model]
            model = cls(**kwargs)
            device = getattr(self.args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            return model

        # --- TAGPPI 任务特殊处理 ---
        if self.args.model == "TAGPPI":
            protein_dim = int(getattr(self.args, 'protein_dim', 1024))
            task_type = getattr(self.args, 'task_type', 'binary')
            # 多标签任务显存压力大，强制用较小的 hidden_dim
            hidden_dim = int(getattr(self.args, 'hidden1', 128))
            if task_type == 'multilabel' or hidden_dim > 128:
                hidden_dim = min(hidden_dim, 128)
            num_classes = int(getattr(self.args, 'num_classes', 2))
            dropout = float(getattr(self.args, 'dropout', 0.5))

            kwargs = {
                'protein_dim': protein_dim,
                'hidden_dim': hidden_dim,
                'dropout': dropout,
                'num_classes': num_classes,
                'task_type': task_type,
            }

            cls = self.model_mapping[self.args.model]
            model = cls(**kwargs)
            device = getattr(self.args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            return model

        # --- PPI_TUnA 任务特殊处理 ---
        if self.args.model == "PPI_TUnA":
            protein_dim = int(getattr(self.args, 'protein_dim', 1024))
            task_type = getattr(self.args, 'task_type', 'binary')
            num_classes = int(getattr(self.args, 'num_classes', 2))
            dropout = float(getattr(self.args, 'dropout', 0.3))
            hidden_dim = int(getattr(self.args, 'hidden1', 256))

            kwargs = {
                'protein_dim': protein_dim,
                'hidden_dim': hidden_dim,
                'dropout': dropout,
                'num_classes': num_classes,
                'task_type': task_type,
            }

            cls = self.model_mapping[self.args.model]
            model = cls(**kwargs)
            device = getattr(self.args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            return model

        # --- MARPPI 任务特殊处理 ---
        if self.args.model == "MARPPI":
            protein_dim = int(getattr(self.args, 'protein_dim', 1024))
            task_type = getattr(self.args, 'task_type', 'binary')
            num_classes = int(getattr(self.args, 'num_classes', 2))
            dropout = float(getattr(self.args, 'dropout', 0.2))
            encoder_dim = int(getattr(self.args, 'hidden1', 512))
            # 多标签 / 大 encoder_dim 显存压力大，强制 cap 到 128
            if task_type == 'multilabel' or encoder_dim > 128:
                encoder_dim = min(encoder_dim, 128)

            kwargs = {
                'protein_dim': protein_dim,
                'encoder_dim': encoder_dim,
                'dropout': dropout,
                'num_classes': num_classes,
                'task_type': task_type,
            }

            cls = self.model_mapping[self.args.model]
            model = cls(**kwargs)
            device = getattr(self.args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            return model

        # --- MAPE_PPI 任务特殊处理 ---
        if self.args.model == "MAPE_PPI":
            protein_dim = int(getattr(self.args, 'protein_dim', 1024))
            task_type = getattr(self.args, 'task_type', 'binary')
            num_classes = int(getattr(self.args, 'num_classes', 2))
            dropout = float(getattr(self.args, 'dropout', 0.2))
            ppi_hidden_dim = int(getattr(self.args, 'hidden1', 512))
            # 多标签 / 大 ppi_hidden_dim 显存压力大，强制 cap 到 128
            if task_type == 'multilabel' or ppi_hidden_dim > 128:
                ppi_hidden_dim = min(ppi_hidden_dim, 128)

            kwargs = {
                'protein_dim': protein_dim,
                'ppi_hidden_dim': ppi_hidden_dim,
                'dropout': dropout,
                'num_classes': num_classes,
                'task_type': task_type,
            }

            cls = self.model_mapping[self.args.model]
            model = cls(**kwargs)
            device = getattr(self.args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            return model

        # --- HIGH_PPI 任务特殊处理 ---
        if self.args.model == "HIGH_PPI":
            protein_dim = int(getattr(self.args, 'protein_dim', 1024))
            task_type = getattr(self.args, 'task_type', 'binary')
            num_classes = int(getattr(self.args, 'num_classes', 2))
            dropout = float(getattr(self.args, 'dropout', 0.3))
            ppi_hidden_dim = int(getattr(self.args, 'hidden1', 256))
            # 多标签 / 大 ppi_hidden_dim 显存压力大，强制 cap 到 128
            if task_type == 'multilabel' or ppi_hidden_dim > 128:
                ppi_hidden_dim = min(ppi_hidden_dim, 128)

            kwargs = {
                'protein_dim': protein_dim,
                'hidden_dim': ppi_hidden_dim,
                'dropout': dropout,
                'num_classes': num_classes,
                'task_type': task_type,
            }

            cls = self.model_mapping[self.args.model]
            model = cls(**kwargs)
            device = getattr(self.args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            return model

        # --- GTB_PPI 任务特殊处理 ---
        if self.args.model == "GTB_PPI":
            protein_dim = int(getattr(self.args, 'protein_dim', 1024))
            task_type = getattr(self.args, 'task_type', 'binary')
            num_classes = int(getattr(self.args, 'num_classes', 2))
            dropout = float(getattr(self.args, 'dropout', 0.3))
            gtb_hidden_dim = int(getattr(self.args, 'hidden1', 256))
            n_estimators = int(getattr(self.args, 'n_estimators', 4))
            max_depth = int(getattr(self.args, 'max_depth', 3))
            # 多标签 / 大 gtb_hidden_dim 显存压力大，强制 cap 到 128
            if task_type == 'multilabel' or gtb_hidden_dim > 128:
                gtb_hidden_dim = min(gtb_hidden_dim, 128)

            kwargs = {
                'protein_dim': protein_dim,
                'hidden_dim': gtb_hidden_dim,
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'dropout': dropout,
                'num_classes': num_classes,
                'task_type': task_type,
            }

            cls = self.model_mapping[self.args.model]
            model = cls(**kwargs)
            device = getattr(self.args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            return model

        # --- GraphPPIS 任务特殊处理 ---
        if self.args.model == "GraphPPIS":
            protein_dim = int(getattr(self.args, 'protein_dim', 1024))
            task_type = getattr(self.args, 'task_type', 'binary')
            num_classes = int(getattr(self.args, 'num_classes', 2))
            dropout = float(getattr(self.args, 'dropout', 0.3))
            hidden_dim = int(getattr(self.args, 'hidden1', 256))
            nlayers = int(getattr(self.args, 'layers', 4))
            # 多标签 / 大 hidden_dim 显存压力大，强制 cap 到 128
            if task_type == 'multilabel' or hidden_dim > 128:
                hidden_dim = min(hidden_dim, 128)

            kwargs = {
                'protein_dim': protein_dim,
                'hidden_dim': hidden_dim,
                'nlayers': nlayers,
                'dropout': dropout,
                'num_classes': num_classes,
                'task_type': task_type,
            }

            cls = self.model_mapping[self.args.model]
            model = cls(**kwargs)
            device = getattr(self.args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            return model

        # --- D_SCRIPT 任务特殊处理 ---
        if self.args.model == "D_SCRIPT":
            protein_dim = int(getattr(self.args, 'protein_dim', 1024))
            task_type = getattr(self.args, 'task_type', 'binary')
            num_classes = int(getattr(self.args, 'num_classes', 2))
            dropout = float(getattr(self.args, 'dropout', 0.5))
            projection_dim = int(getattr(self.args, 'hidden2', 100))
            hidden_dim = int(getattr(self.args, 'hidden1', 50))
            width = int(getattr(self.args, 'layers', 7))
            if task_type == 'multilabel' or projection_dim > 128:
                projection_dim = min(projection_dim, 128)

            kwargs = {
                'protein_dim': protein_dim,
                'projection_dim': projection_dim,
                'dropout': dropout,
                'hidden_dim': hidden_dim,
                'width': width,
                'num_classes': num_classes,
                'task_type': task_type,
            }

            cls = self.model_mapping[self.args.model]
            model = cls(**kwargs)
            device = getattr(self.args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            return model

        # --- CollaPPI 任务特殊处理 ---
        if self.args.model == "CollaPPI":
            protein_dim = int(getattr(self.args, 'protein_dim', 1024))
            task_type = getattr(self.args, 'task_type', 'binary')
            num_classes = int(getattr(self.args, 'num_classes', 2))
            dropout = float(getattr(self.args, 'dropout', 0.2))
            hidden_dim = int(getattr(self.args, 'hidden1', 64))
            num_heads = 2
            # 多标签 / 大 hidden_dim 显存压力大，强制 cap 到 128
            if task_type == 'multilabel' or hidden_dim > 128:
                hidden_dim = min(hidden_dim, 128)

            kwargs = {
                'protein_dim': protein_dim,
                'hidden_dim': hidden_dim,
                'num_heads': num_heads,
                'dropout': dropout,
                'num_classes': num_classes,
                'task_type': task_type,
            }

            cls = self.model_mapping[self.args.model]
            model = cls(**kwargs)
            device = getattr(self.args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            return model

        # --- AdaMBind 任务特殊处理 ---
        if self.args.model == "AdaMBind":
            drug_dim = int(getattr(self.args, 'drug_dim', 1024))
            protein_dim = int(getattr(self.args, 'protein_dim', 1024))
            task_type = getattr(self.args, 'task_type', None)
            if task_type is None:
                task_type = 'regression' if getattr(self.args, 'task', None) == 'dta' else 'classification'
            dropout = float(getattr(self.args, 'dropout', 0.2))
            hidden_dim = int(getattr(self.args, 'hidden1', 1024))
            num_heads = 4

            num_classes = 2 if task_type != 'regression' else 1

            kwargs = {
                'drug_dim': drug_dim,
                'protein_dim': protein_dim,
                'hidden_dim': hidden_dim,
                'num_heads': num_heads,
                'dropout': dropout,
                'num_classes': num_classes,
                'task_type': task_type,
            }

            cls = self.model_mapping[self.args.model]
            model = cls(**kwargs)
            device = getattr(self.args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            return model

        # 分类任务需要 num_classes
        num_classes = int(getattr(self.args, 'num_classes', 0))
        if num_classes <= 0:
            raise ValueError("num_classes 未正确设置；请先在数据加载后赋值。")

        cls = self.model_mapping[self.args.model]
        want = set(signature(cls.__init__).parameters.keys())

        kwargs = {}

        # --- DTI 任务特殊处理 ---
        if self.args.model == "DTI":
            drug_dim = int(getattr(self.args, 'drug_dim', 512))
            protein_dim = int(getattr(self.args, 'protein_dim', 512))
            kwargs['drug_dim'] = drug_dim
            kwargs['protein_dim'] = protein_dim
            kwargs['hidden_dim'] = int(getattr(self.args, 'hidden1', 256))
            kwargs['dropout'] = float(getattr(self.args, 'dropout', 0.3))
            kwargs['num_classes'] = 2
            model = cls(**kwargs)
            device = getattr(self.args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            return model

        # --- 维度：多模态列表 vs 单一维度 ---
        if 'features' in want:
            # 仅在模型真的声明了 'features' 时才传列表（如 MKGFENN 等）
            kwargs['features'] = self.args.features
        if 'feature' in want:
            # 大多数模型只要合并后的维度
            kwargs['feature'] = int(self.args.dimensions)

        # --- 常见参数（有就传）---
        for k in ('hidden1', 'hidden2', 'dropout', 'num_classes',
                'event_sem_dim', 'lambda_align', 'lambda_u_pair',
                'lambda_u_event', 'uniform_t'):
            if k in want and hasattr(self.args, k):
                kwargs[k] = getattr(self.args, k)

        # 有些模型叫 num_relations（通常等于类别数）
        if 'num_relations' in want:
            kwargs['num_relations'] = num_classes

        model = cls(**kwargs)

        # 发送到设备
        device = getattr(self.args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        return model
    
    def load_origin_model(self, ddi_dataset):
        num_classes = int(getattr(self.args, 'num_classes', 0))
        device = getattr(self.args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        if num_classes <= 0:
            raise ValueError("num_classes 未正确设置；请先在数据加载后赋值。")
        if self.args.model == "TIGER":
            model = oritiger(max_layer = 2,
                        num_features_drug = 67,
                        num_nodes = ddi_dataset.data_sta['num_nodes'],
                        num_relations_mol = ddi_dataset.data_sta['num_rel_mol'],
                        num_relations_graph = ddi_dataset.data_sta['num_rel_graph'],
                        output_dim=64,
                        max_degree_graph=ddi_dataset.data_sta['max_degree_graph'],
                        max_degree_node=ddi_dataset.data_sta['max_degree_node'],
                        sub_coeff = 0.2,
                        mi_coeff = 0.5,
                        dropout=0.2,
                        device = device,
                        num_rel = num_classes,
                        args=self.args)
        if self.args.model == "GOGNN":
            model = origognn(args = self.args, 
                        num_features = ddi_dataset.num_features, 
                        nhid = 64, 
                        ddi_nhid = 64, 
                        pooling_ratio = 0.3, 
                        dropout_ratio = 0.3,
                        num_rel = num_classes)
        if self.args.model == "MUFFIN":
            model = orimuffin(args = self.args, 
                        entity_dim = ddi_dataset.entity_dim, 
                        structure_dim = ddi_dataset.structure_dim, 
                        num_rel = num_classes)
        if self.args.model == "MVA":
            model = orimva(args = self.args, 
                        gcn_in_features = 75, 
                        gcn_out_features = 128, 
                        num_rel = num_classes)

        # 发送到设备
        model.to(device)
        return model
