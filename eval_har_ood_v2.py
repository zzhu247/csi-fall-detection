import os, json, sys, torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
sys.path.insert(0, '/home/zhuzih19/csi-project/csibench-official')
sys.path.insert(0, '/home/zhuzih19/csi-project/csi-fall-detection')
from load.supervised.benchmark_loader import load_benchmark_supervised
from models.csibench_models import MLPClassifier, LSTMClassifier, ResNet18Classifier, TransformerClassifier, ViTClassifier, PatchTST, TimesFormer1D

DATA_ROOT = '/home/zhuzih19/data/csi-bench-dataset'
CKPT_DIR  = '/home/zhuzih19/csi-project/csi-fall-detection/results/csibench_official/HumanActivityRecognition'

def build_model(name, nc):
    kw = dict(win_len=500, feature_size=232, num_classes=nc)
    m = {'mlp': MLPClassifier, 'resnet18': ResNet18Classifier, 'vit': ViTClassifier,
         'timesformer1d': TimesFormer1D}[name](**kw) if name in ('mlp','resnet18','vit') \
        else PatchTST(depth=6, **kw) if name == 'patchtst' \
        else TimesFormer1D(depth=6, **kw) if name == 'timesformer1d' \
        else LSTMClassifier(feature_size=232, num_classes=nc) if name == 'lstm' \
        else TransformerClassifier(feature_size=232, num_classes=nc)
    return m

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    p, l = [], []
    for batch in loader:
        if batch[0].shape[0] == 0: continue
        csi, y = batch
        p.append(model(csi.to(device)).argmax(1).cpu())
        l.append(y)
    p, l = torch.cat(p).numpy(), torch.cat(l).numpy()
    return float((p==l).mean()), float(f1_score(l, p, average='weighted', zero_division=0))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load all splits using official dataset
print("Loading data via official BenchmarkCSIDataset...")
data = load_benchmark_supervised(
    dataset_root=DATA_ROOT,
    task_name='HumanActivityRecognition',
    batch_size=128,
    num_workers=4,
    test_splits='all',
    pin_memory=False,
)
loaders   = data['loaders']
nc        = data['num_classes']
label_mapper = data['label_mapper']
print(f"num_classes={nc}, label_map={label_mapper.label_to_idx}")
print(f"Available splits: {list(loaders.keys())}\n")

for mn in ['mlp','lstm','resnet18','transformer','vit','patchtst','timesformer1d']:
    cp = f'{CKPT_DIR}/{mn}/params_828dca1639/best_model.pt'
    if not os.path.exists(cp): print(f'SKIP {mn}'); continue
    ck = torch.load(cp, map_location=device)
    m  = build_model(mn, nc).to(device)
    m.load_state_dict(ck.get('model_state_dict', ck.get('model_state', ck)))
    print(f'[{mn}]')
    for sn in list(loaders.keys()):
        if sn not in loaders: print(f'  {sn}: not found'); continue
        acc, f1 = evaluate(m, loaders[sn], device)
        print(f'  {sn:25s} acc={acc*100:.2f}% f1={f1*100:.2f}%')
