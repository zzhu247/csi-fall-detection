# config.py
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = "/home/zhuzih19/data/csi-bench-dataset"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Model
IN_CHANNELS = 1
IMG_H       = 232
IMG_W       = 500
PATCH_H     = 8
PATCH_W     = 25
D_MODEL     = 128
D_FF        = 512
N_HEADS     = 4
N_LAYERS    = 12
TARGET_LAYERS = [1, 4, 8, 12]
NUM_CLASSES = 2

# Training
BATCH_SIZE  = 16
LR          = 3e-5
NUM_EPOCHS  = 10
NUM_WORKERS = 0


OOD_SPLITS = {'test_id', 'test_cross_device', 'test_cross_environment', 'test_cross_user'}