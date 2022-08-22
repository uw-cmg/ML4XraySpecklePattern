"""
Using Keras to train Resnet50 model for XPCS data

"""
# import standarded package
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils import plot_model
from keras.models import model_from_json
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np
import argparse
import os

# import user defined packages
from utils import datasets
#from utils import models
from utils import miscFuncs
from utils import resnet

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
  help="path to input dataset of micelles and speckle pattern images")
ap.add_argument("-s", "--savedResults", type=str, required=True,
  help="path to save results of both model and weights")
ap.add_argument("-l", "--trainingLog", type=str, required=True,
  help="path to save training logs and metrics curves")
args = vars(ap.parse_args())

# create results file when needed
if not os.path.exists(args["savedResults"]):
    os.makedirs(args["savedResults"])

if not os.path.exists(args["trainingLog"]):
    os.makedirs(args["trainingLog"])


lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('resnet50_trainingLog.csv')

batch_size = 8 
nb_classes = 3
nb_img_per_class = 1000
nb_epoch = 2
data_augmentation = True

# input image dimensions
img_rows, img_cols = 513, 513
# The images are duplicated to fit RGB channels.
img_channels = 3

print("[INFO] loading Spekcle Pattern Data and Label")
X_raw , y_raw= datasets.load_data_std(args["dataset"], nb_classes, nb_img_per_class, img_rows, img_cols)

X = np.array(X_raw).astype('float32')
print("[INFO] Spekcle Pattern Data Shape")
print(X.shape)
# Convert class vectors to one-hot class matrices.
y = np_utils.to_categorical(y_raw, nb_classes)

# Split dataset into train_valid and test
X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle= True)
# Split train_valid into train and valid
X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.11, random_state=42, shuffle= True)

# Saving Testing data
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("[INFO] build model...")
model = resnet.ResnetBuilder.build_resnet_50((img_channels, img_rows, img_cols), nb_classes)
model.summary()
plot_model(model, to_file='model.png')

print("[INFO] complie model...")
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print("[INFO] training model...")
history = model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_valid, y_valid),
          shuffle=True,
          callbacks=[lr_reducer, early_stopper, csv_logger])
# Plot the training Processing
# https://keras.io/visualization/
# list all data in history
print(history.history.keys())

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig(args["trainingLog"]+"/loss.png", dpi=150)
plt.close()

# Plot training & validation MAE values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'],loc='upper right')
#plt.show()
plt.savefig(args["trainingLog"]+"/ACC.png", dpi=150)
plt.close()

print("[INFO] saving trained model...")
# Saving Trained model
# serialize model to JSON
model_json = model.to_json()
with open(args["savedResults"] + "/model_structure.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(args["savedResults"]+"/model_weights.h5")
print("Saved model to disk")

# Testing Performance
print("[INFO] Testing model...")

#Confution Matrix and Classification Report
predicted = model.predict(X_test)
np.save("y_pred_test_raw.npy", predicted)
y_pred = np.argmax(predicted, axis=1)
np.save("y_pred_test.npy", y_pred)

y_gt = np.argmax(y_test, axis=1)
np.save("y_gt_test.npy", y_gt)

print(accuracy_score(y_gt, y_pred))

# Saving predict and gt results
with open("gt_pred_allResults.csv",'w') as predFile:
  predFile.write("%s, %s ,%s, %s\n"%("indx", "gt_label","pred_label", "nb_beads"))
  assert len(y_pred) == len(y_gt)
  for ind in range(len(y_gt)):
    predFile.write("%d, %d ,%d, %d\n"%(ind, y_gt[ind], y_pred[ind], y_gt[ind]+2))

# generating class labels
class_names = []
for ii in range(nb_classes):
  if ii+2 < 10:
    class_names.append("Beads_0" + str(ii+2))
  else:
    class_names.append("Beads_" + str(ii+2))

print('Confusion Matrix')
confusionMatrix = confusion_matrix(y_gt, y_pred)
print(confusion_matrix(y_gt, y_pred))
miscFuncs.plot_confusion_matrix(confusionMatrix, class_names, normalize=False)

print('Classification Report')
print(classification_report(y_gt, y_pred, target_names=class_names))

"""
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True,
              callbacks=[lr_reducer, early_stopper, csv_logger])
else:
    print("E")
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        steps_per_epoch=X_train.shape[0] // batch_size,
                        validation_data=(X_test, Y_test),
                        epochs=nb_epoch, verbose=1, max_q_size=100,
                        callbacks=[lr_reducer, early_stopper, csv_logger])
"""
