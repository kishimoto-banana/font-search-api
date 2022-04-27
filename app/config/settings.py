import pickle

NUM_TOP_K = 5
PATCH_SIZE = 105
NUM_PATCHES = 5
MODEL_PATH = "models/font365_vgg16.pt"

try:
    with open("models/meta.pkl", mode="rb") as f:
        FONT_LABEL_TO_META = pickle.load(f)
except FileNotFoundError:
    print("testのためだがどうにかしたい")
