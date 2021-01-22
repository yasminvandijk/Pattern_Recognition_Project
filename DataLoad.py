#Load in data
import numpy as np
from skimage import color, exposure, transform

NUM_CLASSES = 43
IMG_SIZE = 48

## preprocess images
def preprocess_img(img):
    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    # central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0] // 2, img.shape[1] // 2
    img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
              centre[1] - min_side // 2:centre[1] + min_side // 2,
              :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # roll color axis to axis 0
    img = np.rollaxis(img, -1)

    return img

##import training images
from skimage import io
import os
import glob


def get_class(img_path):
    return int(img_path.split(os.sep)[-2])

def load_data():
    root_dir = 'GTSRB' + os.sep + 'Final_Training' + os.sep + 'Images' + os.sep

    imgs = []
    labels = []

    all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))

    np.random.shuffle(all_img_paths)
    for img_path in all_img_paths:
        img = preprocess_img(io.imread(img_path))
        label = get_class(img_path)
        imgs.append(img)
        labels.append(label)

    X = np.array(imgs, dtype='float32')
    # Make one hot targets
    Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]

    return (X,Y)

