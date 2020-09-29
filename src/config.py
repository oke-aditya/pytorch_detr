TRAIN_CSV_PATH = "df_train.csv"
VALIDATION_CSV_PATH = "df_val.csv"
IMAGE_DIR = "/content/drive/My Drive/VAUV_Dataset/final_data/youtube/2017_VID_5FPS/"
BACKBONE = "detr_resnet50"
PRETRAINED = True

IMG_HEIGHT = 512
IMG_WIDTH = 512
NUM_QUERIES = 10
TARGET = "target"
# BATCH_SIZE = 32
TRAIN_BATCH_SIZE = 2
VALID_BATCH_SIZE = 2
TRAIN_WORKERS = 4
LEARNING_RATE = 1e-3
EPOCHS = 3
NUM_CLASSES = 4
DETECTION_THRESHOLD = 0.25
NULL_CLASS_COEF = 0.5

MODEL_SAVE_PATH = "detr_{}.pt".format(BACKBONE)
