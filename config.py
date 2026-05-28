# config.py

DATA_ROOT = "/Volumes/csi/default/raw_data/csi-bench-dataset/csi-bench-dataset"

# Model
IN_CHANNELS = 1
IMG_H       = 232
IMG_W       = 500
PATCH_H     = 8
PATCH_W     = 25
D_MODEL     = 128
D_FF        = 512
N_HEADS     = 4
N_LAYERS    = 4
NUM_CLASSES = 2

# Training
BATCH_SIZE  = 16
LR          = 3e-5
NUM_EPOCHS  = 10
NUM_WORKERS = 0