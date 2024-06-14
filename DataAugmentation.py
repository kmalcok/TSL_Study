from __future__ import print_function
import tensorflow as tf
from tensorflow.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input/tsl finger spelling/"))

datagen = ImageDataGenerator(
        rotation_range=20,
        rescale=1./255,
        horizontal_flip=True,
        fill_mode='nearest')

for k in os.scandir('../input/tsl finger spelling/Images'):
    print(k)
    print(os.path.relpath(k).split('/')[4].split(' ')[0])
    prefix = os.path.relpath(k).split('/')[4].split(' ')[0]

    img = load_img(os.path.relpath(k))  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    i = 0

    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=out_Path, save_prefix=prefix, save_format='png'):
        i += 1

        if i > 4:
            break
