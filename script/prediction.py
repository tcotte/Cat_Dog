# Librairies
print("Load Libraries")
import os

import numpy as np
import pandas as pd

import tensorflow.keras.preprocessing.image as kpi
import tensorflow.keras.models as km
from tensorflow import keras

from tensorflow.python.client import device_lib

MODE = "GPU" if "GPU" in [k.device_type for k in device_lib.list_local_devices()] else "CPU"
print(MODE)

## Argument
import argparse

# TODO Write here the parameters that can be given as inputs to the algorithm.
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--data-dir', type=str, default='data')
parser.add_argument('--results-dir', type=str, default='results')
parser.add_argument('--model-dir', type=str, default='model')
parser.add_argument('--epochs', type=int, default=10)
args = parser.parse_args()

# TODO Define generator.


## Data Generator
img_width = 150
img_height = 150

data_dir_test = args.data_dir + '/test'
N_test = len(os.listdir(data_dir_test + "/test"))

test_datagen = kpi.ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(directory=data_dir_test,
                                                  # batch_size=args.batch_size,
                                                  class_mode='binary',
                                                  target_size=(img_width, img_height),
                                                  shuffle=False)

## Download model
# Todo Download model saved in learning script.
args_str = "epochs_%d_batch_size_%d" % (args.epochs, args.batch_size)

model = km.load_model(args.model_dir + "/" + args_str + ".h5")

## Prediction
# Todo Generate prediction.

pred = model.predict_generator(test_generator, steps=len(test_generator), verbose=1)
cl = np.round(pred)
filenames = test_generator.filenames


## Save prediction in csv
# TODO Save the results in a csv file.

def CatOrDog(x):
    if x == [0]:
        return "Cat"
    else:
        return "Dog"


animal = list(map(CatOrDog, cl))

filenames = list(map(lambda x: x[5:], filenames))

pred_df = pd.DataFrame({"predictions": cl[:, 0], "animal": animal, "filename": filenames})

## Save prediction in csv
pred_file = args.results_dir+'/predictions.csv'
with open(pred_file, mode='w') as f:
    pred_df.to_csv(f)
