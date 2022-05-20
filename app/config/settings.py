import pickle

NUM_TOP_K = 5
PATCH_SIZE = 105
PATCH_MARGIN = 3
MODEL_PATH = "models/char_font365_vgg16.pt"

try:
    with open("models/meta.pkl", mode="rb") as f:
        FONT_LABEL_TO_META = pickle.load(f)
except FileNotFoundError:
    print("testのためだがどうにかしたい")
