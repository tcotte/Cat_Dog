# Librairies
print("Load Libraries")
import time
import pickle
import os
import tensorflow.keras.preprocessing.image as kpi
import tensorflow.keras.layers as kl
import tensorflow.keras.models as km
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import pandas as pd
import tensorflow.keras.layers as kl
import tensorflow.keras.models as km

from tensorflow.python.client import device_lib

MODE = "GPU" if "GPU" in [k.device_type for k in device_lib.list_local_devices()] else "CPU"
print(MODE)

## Argument
import argparse

# TODO Write here the parameters that can be given as inputs to the algorithm.
# data-dir / epochs / batch-size / results-dir / model-dir
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default='data')
parser.add_argument('--results-dir', type=str, default='results')
parser.add_argument('--model-dir', type=str, default='model')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=256)
args = parser.parse_args()

## Data Generator

img_width = 150
img_height = 150

N_train = len(os.listdir(args.data_dir + "/train/cats/")) + len(os.listdir(args.data_dir + "/train/dogs/"))
N_val = len(os.listdir(args.data_dir + "/validation/cats/")) + len(os.listdir(args.data_dir + "/validation/dogs/"))
print("%d   %d" % (N_train, N_val))

# TODO Write here code to generate obh train and validation generator
train_datagen = kpi.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(directory=args.data_dir + '/train',
                                                    batch_size=args.batch_size,
                                                    class_mode='binary',
                                                    target_size=(img_width, img_height))

valid_datagen = kpi.ImageDataGenerator(rescale=1./255)

validation_generator = valid_datagen.flow_from_directory(directory=args.data_dir + '/validation',
                                                         batch_size=args.batch_size,
                                                         class_mode='binary',
                                                         target_size=(img_width, img_height))

## Model
# TODO Write a simple convolutional neural network

# Create model
# model_conv = Sequential()
#
# model_conv.add(Conv2D(args.batch_size, kernel_size=30, activation='relu', input_shape=(img_width, img_height, 3)))
# model_conv.add(Conv2D(32, kernel_size=3, activation='relu'))
# model_conv.add(Flatten())
# model_conv.add(Dense(1, activation='sigmoid'))

model_conv = km.Sequential()
model_conv.add(kl.Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3), data_format="channels_last"))
model_conv.add(kl.Activation('relu'))
model_conv.add(kl.MaxPooling2D(pool_size=(2, 2)))

model_conv.add(kl.Conv2D(32, (3, 3)))
model_conv.add(kl.Activation('relu'))
model_conv.add(kl.MaxPooling2D(pool_size=(2, 2)))

model_conv.add(kl.Conv2D(64, (3, 3)))
model_conv.add(kl.Activation('relu'))
model_conv.add(kl.MaxPooling2D(pool_size=(2, 2)))

model_conv.add(kl.Flatten())  # this converts our 3D feature maps to 1D feature vectors
model_conv.add(kl.Dense(64))
model_conv.add(kl.Activation('relu'))
model_conv.add(kl.Dropout(0.5))
model_conv.add(kl.Dense(1))
model_conv.add(kl.Activation('sigmoid'))

# Compile model

# compile model using accuracy to measure model performance
model_conv.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

## Learning

print("Start Learning")
ts = time.time()
history = model_conv.fit(train_generator, steps_per_epoch=N_train // args.batch_size, epochs=args.epochs,
                                   validation_data=validation_generator, validation_steps=N_val // args.batch_size)
te = time.time()
t_learning = te - ts

## Test
# TODO Calculez l'accuracy de votre modèle sur le jeu d'apprentissage et sur le jeu de validation.

print("Start predicting")
ts = time.time()
score_train = model_conv.evaluate_generator(train_generator, N_train / args.batch_size, verbose=1)
score_val = model_conv.evaluate_generator(validation_generator, N_val / args.batch_size, verbose=1)
te = time.time()
t_prediction = te - ts

args_str = "epochs_%d_batch_size_%d" % (args.epochs, args.batch_size)

## Save Model

# TODO Save model in model folder

model_conv.save(args.model_dir + "/" + args_str + ".h5")
print('saving the model...')

## Save results

## TODO Save results (learning time, prediction time, train and test accuracy, history.history object) in result folder

d = {'learning_time': t_learning, 'prediction_time': t_prediction, 'accuracy_train': score_train, 'accuracy_test' : score_val }
results_df = pd.DataFrame(data=d)
results_file = args.results_dir+'/results.csv'
with open(results_file, mode='w') as f:
    results_df.to_csv(f)

# convert the history.history dict to a pandas DataFrame:
hist_df = pd.DataFrame(history.history)
# and save to csv:
hist_csv_file = args.results_dir+'/history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)