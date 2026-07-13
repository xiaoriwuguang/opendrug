import os
import argparse
import torch
import random
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

set_random_seed(1, deterministic=True)
CODE_DIR = os.path.dirname(os.path.abspath(__file__))

def settings():
    parser = argparse.ArgumentParser()

    # ---------------- 基础训练参数 ----------------
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Force use CPU')
    parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'cpu'],
                        default='cuda', help="Device: 'auto' -> cuda if available else cpu")
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=150)

    # 输出文件目录
    default_fig_dir = os.path.join(CODE_DIR, '..', 'results')
    os.makedirs(default_fig_dir, exist_ok=True)

    # 模态切分（可选）
    parser.add_argument('--modal_splits', type=str, default=None,
                        help='各模态维度, 逗号分隔, 之和需等于 args.dimensions, 例如 "1024,768,256,128"')

    # ---------------- 模型/任务选择 ----------------
    parser.add_argument('--model', type=str,
                        choices=['MRCGNN','GOGNN','MVA','DeepDDI','DDKG',
                                 'SumGNN','PHGLDDI','ExDDI','CASTER','MKGFENN',
                                 'KGE_NFM', 'MGraphDTA', 'MMD_DTA', 'RSGCL_DTI',
                                 'GraphDTA', 'EviDTI', 'DTIAM', 'DrugBAN', 'ColdstartCPI', 'AdaMBind',
                                 'DL_PPI', 'TAGPPI', 'PPI_TUnA', 'MARPPI', 'MAPE_PPI',
                                 'HIGH_PPI', 'GTB_PPI', 'GraphPPIS', 'D_SCRIPT', 'CollaPPI'
                                 ],
                        default='PHGLDDI')
    parser.add_argument('--network_ratio', type=float, default=0.1)
    parser.add_argument('--loss_ratio1', type=float, default=1.0)
    parser.add_argument('--loss_ratio2', type=float, default=0.05)
    parser.add_argument('--loss_ratio3', type=float, default=0.1)
    parser.add_argument('--hidden1', type=int, default=512)
    parser.add_argument('--hidden2', type=int, default=256)
    parser.add_argument('--layers', type=int, default=2, help='Number of DNN layers for KGE_NFM')
    parser.add_argument('--kge_nfm_variant', type=str, default='standard',
                       choices=['standard', 'bilinear', 'attention'],
                       help='KGE_NFM model variant')
    parser.add_argument('--mgraphdta_variant', type=str, default='embedding',
                       choices=['embedding', 'graph'],
                       help='MGraphDTA model variant')

    # DrugBAN 特定参数
    parser.add_argument('--ban_heads', type=int, default=2,
                       help='Number of attention heads in BAN layer')
    parser.add_argument('--ban_variant', type=str, default='standard',
                       choices=['standard', 'enhanced'],
                       help='DrugBAN variant')

    # ColdstartCPI 特定参数
    parser.add_argument('--unify_num', type=int, default=256,
                       help='Unified representation dimension for ColdstartCPI')
    parser.add_argument('--max_drug_seq', type=int, default=100,
                       help='Max drug sequence length for ColdstartCPI')
    parser.add_argument('--max_protein_seq', type=int, default=1000,
                       help='Max protein sequence length for ColdstartCPI')

    # DL-PPI 特定参数
    parser.add_argument('--gin_in_feature', type=int, default=256,
                       help='GIN input feature dimension for DL-PPI')
    parser.add_argument('--gin_layers', type=int, default=1,
                       help='Number of GIN layers for DL-PPI')
    parser.add_argument('--gin_pool_size', type=int, default=3,
                       help='Pooling window size for DL-PPI CNN')
    parser.add_argument('--dlppi_fusion', type=str, default='NTN',
                       choices=['NTN', 'concat', 'mult'],
                       help='Feature fusion method for DL-PPI (NTN/concat/mult)')

    # GTB-PPI 特定参数
    parser.add_argument('--n_estimators', type=int, default=4,
                       help='Number of GTB blocks (simulates n_estimators in Gradient Tree Boosting)')
    parser.add_argument('--max_depth', type=int, default=3,
                       help='Depth of each GTB residual block (simulates max_depth in Gradient Tree Boosting)')

    # ---------------- 数据集/模态 ----------------
    parser.add_argument('--features', type=int, nargs='+', default=[300, 320, 512, 320, 768])
    parser.add_argument('--dimensions', type=int, default=512)
    parser.add_argument('--num_classes', type=int, default=-1)
    parser.add_argument('--task', type=str, choices=['train_xxxx', 'dta', 'dti', 'ppi_b', 'ppi_m'], default='train_xxxx',
                        help='train_xxxx: DDI任务, dta: DTA回归任务, dti: DTI分类任务, ppi_b: PPI二分类, ppi_m: PPI多标签')
    parser.add_argument('--matrix', type=str,
                        choices=['binary','zhangddi','ChCh-Miner',
                                 'multi','zeroddi','Dengs','Ryus',
                                 'multilabel','twosides',
                                 'dti_od', 'bindingdb', 'BIOSNAP',
                                 'dta_od', 
                                 'ppi_binary_od', 'metabolism', 'neurodegenerative'
                                 'ppi_multilabel_od', 'SHS148k',],
                        default='Ryus')
    parser.add_argument('--modality', type=str, nargs='+',
                        choices=['drug_smiles','drug_sequence','drug_3d','drug_mechanism','drug_text',
                                 'protein_sequence','protein_structure','protein_text'],
                        default=['drug_smiles','drug_sequence','drug_3d','drug_mechanism','drug_text',
                                 'protein_sequence','protein_structure','protein_text'])

    parser.add_argument('--matrix_dir',    type=str, default=os.path.join(CODE_DIR,'..','datasets','matrix'))
    parser.add_argument('--embedding_dir', type=str, default=os.path.join(CODE_DIR,'..','datasets','emb'))
    parser.add_argument('--origin', type=bool, default=False, help='是否使用原始模型')
    parser.add_argument('--general', type=bool, default=False, help='是否进行泛化实验')

    # 噪声
    parser.add_argument('--noise_std', type=float, default=0.0, help='输入特征高斯噪声 σ')
    parser.add_argument('--noise_ratio', type=float, default=0.0, help='训练集标签加噪比例')
    parser.add_argument('--noise_type', type=str, default='symmetric',
                       choices=['symmetric', 'asymmetric'],
                       help='标签噪声类型: symmetric(对称翻转) 或 asymmetric(仅翻转正类)')
    parser.add_argument('--noise_edge', type=float, default=0.0,
                       help='图结构边噪声比例: 按比例随机添加/删除训练集中的边')

    # 稀疏性（可选）
    parser.add_argument('--sparse_drop_rate', type=float, default=0.0)      # 特征随机置零比例
    parser.add_argument('--sparse_sample_rate', type=float, default=0.0)    # 训练集标签采样比例

    # Zero-shot
    parser.add_argument('--event_sem_path', type=str, default=None, help='K×d_e 的 .npy/.csv；缺省为 one-hot')
    parser.add_argument('--zs_protocol', type=str, choices=['none','CZSL','GZSL'], default='none')
    parser.add_argument('--zs_ratio', type=float, default=0.3)
    parser.add_argument('--zs_seed', type=int, default=1)


    # 对齐损失
    parser.add_argument('--lambda_align', type=float, default=1.0)
    parser.add_argument('--lambda_u_pair', type=float, default=0.1)
    parser.add_argument('--lambda_u_event', type=float, default=0.1)
    parser.add_argument('--uniform_t', type=float, default=2.0)

    # ---------- 先解析参数 ----------
    args = parser.parse_args()

    # ---------- 设备选择规范化 ----------
    if args.no_cuda:
        args.device = 'cpu'
    elif args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 若用户强制 --device=cuda 但系统不可用，则降级
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[WARN] --device=cuda 但 PyTorch 未检测到 CUDA，退回 CPU。")
        args.device = 'cpu'
    args.cuda = (args.device == 'cuda')

    # ---------- 路径/映射 ----------
    # 输出文件（为每个模型单独的子目录）
    model_fig_dir = os.path.join(default_fig_dir, args.model)
    os.makedirs(model_fig_dir, exist_ok=True)
    args.out_file = os.path.join(
        model_fig_dir,
        f'{args.model}_{args.task}_{args.matrix}_results.txt'
    )

    args.embedding_map = {
        'drug_smiles':      os.path.join(args.embedding_dir, 'drug_smiles_emb.pt'),
        'drug_sequence':    os.path.join(args.embedding_dir, 'drug_sequence_emb.pt'),
        'drug_3d':          os.path.join(args.embedding_dir, 'drug_3d_emb.pt'),
        'drug_mechanism':   os.path.join(args.embedding_dir, 'drug_mechanism_emb.pt'),
        'drug_text':        os.path.join(args.embedding_dir, 'drug_text_emb.pt'),
        'protein_sequence':  os.path.join(args.embedding_dir, 'protein_sequence_embeddings.pt'),
        'protein_structure': os.path.join(args.embedding_dir, 'protein_structure_embeddings.pt'),
        'protein_text':     os.path.join(args.embedding_dir, 'protein_text_embeddings.pt'),
    }
    args.matrix_map = {
        'binary':     os.path.join(args.matrix_dir, 'ddi_od.csv'),
        'ChCh-Miner': os.path.join(args.matrix_dir, 'ChCh-Miner.csv'),
        'zhangddi':   os.path.join(args.matrix_dir, 'zhangddi.csv'),
        'multi':      os.path.join(args.matrix_dir, 'ddi_multi_od.csv'),
        'Dengs':      os.path.join(args.matrix_dir, 'Dengs.csv'),
        'Ryus':       os.path.join(args.matrix_dir, 'Ryus.csv'),
        'zeroddi':    os.path.join(args.matrix_dir, 'zeroddi.csv'),
        'multilabel': os.path.join(args.matrix_dir, 'ddi_multilabel_od.csv'),
        'twosides':   os.path.join(args.matrix_dir, 'twosides.csv'),
        # DTA 任务
        'dta_od':        os.path.join(args.matrix_dir, 'dta_od.tsv'),
        # DTI 任务
        'dti_od':            os.path.join(args.matrix_dir, 'dti_od.tsv'),
        'bindingdb':      os.path.join(args.matrix_dir, 'bindingdb.tsv'),
        'BIOSNAP':        os.path.join(args.matrix_dir, 'BIOSNAP.tsv'),
        # PPI 任务
        'ppi_binary_od':     os.path.join(args.matrix_dir, 'ppi_od.tsv'),
        'metabolism':      os.path.join(args.matrix_dir, 'metabolism.tsv'),
        'neurodegenerative': os.path.join(args.matrix_dir, 'neurodegenerative.tsv'),
        # PPI 多标签
        'SHS148k':        os.path.join(args.matrix_dir, 'SHS148k.tsv'),
        'ppi_multilabel_od': os.path.join(args.matrix_dir, 'ppi_multilabel_od.tsv'),

       
    }

    # 构建嵌入路径列表
    args.embedding_path = [args.embedding_map[m] for m in args.modality]

        # DTA 任务特殊处理：解析药物嵌入和蛋白质嵌入
    if args.task == 'dta':
        drug_mods = [m for m in args.modality if m.startswith('drug_')]
        protein_mods = [m for m in args.modality if m.startswith('protein_')]
        args.drug_embedding_paths = [args.embedding_map[m] for m in drug_mods] if drug_mods else []
        args.protein_embedding_paths = [args.embedding_map[m] for m in protein_mods] if protein_mods else []
        print(f"[INFO] DTA task: drug modalities={drug_mods}, protein modalities={protein_mods}")
        # DTI 任务特殊处理：解析药物嵌入和蛋白质嵌入
    elif args.task == 'dti':
        drug_mods = [m for m in args.modality if m.startswith('drug_')]
        protein_mods = [m for m in args.modality if m.startswith('protein_')]
        args.drug_embedding_paths = [args.embedding_map[m] for m in drug_mods] if drug_mods else []
        args.protein_embedding_paths = [args.embedding_map[m] for m in protein_mods] if protein_mods else []
        print(f"[INFO] DTI task: drug modalities={drug_mods}, protein modalities={protein_mods}")
    # PPI 任务特殊处理：只用蛋白质嵌入
    elif args.task in ('ppi_b', 'ppi_m'):
        protein_mods = [m for m in args.modality if m.startswith('protein_')]
        args.protein_embedding_paths = [args.embedding_map[m] for m in protein_mods] if protein_mods else []
        args.ppi_type = 'binary' if args.task == 'ppi_b' else 'multilabel'
        print(f"[INFO] PPI task: protein modalities={protein_mods}, ppi_type={args.ppi_type}")
    # DDI 任务特殊处理：只用药物嵌入
    else:
        drug_mods = [m for m in args.modality if m.startswith('drug_')]
        args.drug_embedding_paths = [args.embedding_map[m] for m in drug_mods] if drug_mods else []
        args.protein_embedding_paths = []
        args.embedding_path = args.drug_embedding_paths
        print(f"[INFO] DDI task: drug modalities={drug_mods}")
    # --matrix 未指定时自动推导
    if args.matrix is None:
        task_matrix_map = {
            'train_xxxx': 'binary',
            'dta': 'dta_od',
            'dti': 'dti_od',
            'ppi_b': 'ppi_binary_od',
            'ppi_m': 'ppi_multilabel_od',
        }
        args.matrix = task_matrix_map.get(args.task, 'binary')
        print(f"[INFO] matrix auto-set to: {args.matrix}")
    args.matrix_path = args.matrix_map[args.matrix]
    args.oridata_path = os.path.join(CODE_DIR,'..','datasets','data')
    args.oriSmiles_path = os.path.join(CODE_DIR,'..','datasets','data','id_smiles.csv')
    args.oriKG_path = os.path.join(CODE_DIR,'..','datasets','data','kgnet.tsv')
    args.code_dir = CODE_DIR

    # 小提示：打印一下设备/可见卡，便于你检查
    if args.cuda:
        print(f"[INFO] device=cuda | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','ALL')} "
              f"| cuda_count={torch.cuda.device_count()}")
    else:
        print("[INFO] device=cpu")

    return args
