import os
from PIL import Image
import numpy as np

SAVE_PATH = '../data/img/'

def load_images(path=SAVE_PATH,
                nsamples=None,
                size=(64, 64)):

    files = os.listdir(path)
    if nsamples is None:
        nsamples = len(files)
    count = 0
    images = []

    for i, filename in enumerate(files):
        try:
            img = Image.open(os.path.join(path, filename)).resize(size)

            img = img.convert("RGB")
            img = np.asarray(img, dtype=np.float32) / 255
            img = img[:, :, :3]

            images.append(img)
            count += 1
        except:
            print(f"File \"{filename}\" was excluded from images.")

        if count > nsamples:
            break

    return np.asarray(images)
