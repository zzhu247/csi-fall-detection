import os, json, sys, torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
sys.path.insert(0, '/home/zhuzih19/csi-project/csi-fall-detection')
import config
from data.dataset import load_and_normalize_csi
from models.csibench_models import MLPClassifier, LSTMClassifier, ResNet18Classifier, TransformerClassifier, ViTClassifier, PatchTST, TimesFormer1D

DATA_ROOT  = config.DATA_ROOT
META_PATH  = f'{DATA_ROOT}/Multitask/HumanActivityRecognition/metadata/sample_metadata.csv'
SPLITS_DIR = f'{DATA_ROOT}/Multitask/HumanActivityRecognition/splits'
CKPT_DIR   = '/home/zhuzih19/csi-project/csi-fall-detection/results/csibench_official/HumanActivityRecognition'

class HARDataset(Dataset):
    def __init__(self, meta_df, label_map):
        self.meta = meta_df.reset_index(drop=True)
        self.label_map = label_map
        meta_dir = os.path.dirname(META_PATH)
        self.meta['h5_path'] = self.meta['file_path'].apply(lambda p: os.path.normpath(os.path.join(meta_dir, p)))
    def __len__(self): return len(self.meta)
    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        csi = load_and_normalize_csi(row['h5_path'])
        csi = (csi - csi.mean()) / (csi.std() + 1e-8)
        csi = torch.tensor(csi, dtype=torch.float32).unsqueeze(0)  # [1, 232, 500]
        csi = csi.permute(0, 2, 1)  # -> [1, 500, 232] to match official training
        return csi, torch.tensor(self.label_map[row['label']], dtype=torch.long)

def load_split(name, meta):
    with open(f'{SPLITS_DIR}/{name}.json') as f: ids = set(json.load(f))
    return meta[meta['id'].isin(ids)].reset_index(drop=True)

def build_model(name, nc):
    kw = dict(win_len=500, feature_size=232, num_classes=nc)
    return {'mlp': MLPClassifier(**kw), 'resnet18': ResNet18Classifier(**kw), 'vit': ViTClassifier(depth=6, **kw), 'patchtst': PatchTST(depth=6, **kw), 'timesformer1d': TimesFormer1D(depth=6, **kw), 'lstm': LSTMClassifier(feature_size=232, num_classes=nc), 'transformer': TransformerClassifier(feature_size=232, num_classes=nc)}[name]

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    p, l = [], []
    for csi, y in loader:
        p.append(model(csi.to(device)).argmax(1).cpu()); l.append(y)
    p, l = torch.cat(p).numpy(), torch.cat(l).numpy()
    return float((p==l).mean()), float(f1_score(l, p, average='weighted', zero_division=0))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
meta = pd.read_csv(META_PATH)
train_df = load_split('train_id', meta)
label_map = {l: i for i, l in enumerate(sorted(train_df['label'].unique(), key=str))}
print('label_map:', label_map)
splits = {s: load_split(s, meta) for s in ['test_id','test_cross_device','test_cross_env','test_cross_user']}

for mn in ['mlp','lstm','resnet18','transformer','vit','patchtst','timesformer1d']:
    cp = f'{CKPT_DIR}/{mn}/params_828dca1639/best_model.pt'
    if not os.path.exists(cp): print(f'SKIP {mn}'); continue
    ck = torch.load(cp, map_location=device)
    m = build_model(mn, len(label_map)).to(device)
    m.load_state_dict(ck.get('model_state_dict', ck.get('model_state', ck)))
    print(f'\n[{mn}]')
    for sn, df in splits.items():
        acc, f1 = evaluate(m, DataLoader(HARDataset(df, label_map), batch_size=128, shuffle=False, num_workers=4), device)
        print(f'  {sn:25s} acc={acc*100:.2f}% f1={f1*100:.2f}%')
