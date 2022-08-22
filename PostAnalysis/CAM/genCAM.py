import ast
import glob
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import scipy
from utils import resnet
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.preprocessing import image
from vis.utils import utils
from keras.activations import linear
from vis.visualization import visualize_cam
from matplotlib.font_manager import _rebuild

batch_size = 10 
nb_classes = 19
nb_img_per_class = 1000
nb_epoch = 2
data_augmentation = True

# input image dimensions
img_rows, img_cols = 201, 201
# The images are duplicated to fit RGB channels.
img_channels = 3

# generating class labels
classlabel = []
for ii in range(nb_classes):
    if ii+2 < 10:
        classlabel.append("Beads_0" + str(ii+2))
    else:
        classlabel.append("Beads_" + str(ii+2))

model = resnet.ResnetBuilder.build_resnet_50((img_channels, img_rows, img_cols), nb_classes)
model.summary()

# loading weights of the trained model
from keras.models import model_from_json
# load json and create model
json_file = open('saved/model_structure.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("saved/model_weights.h5")
print("Loaded model from disk")

# loading data
Xtest = np.load("X_test.npy")
Ytest = np.load("y_gt_test.npy")
Ypred = np.load("y_pred_test_raw.npy")


# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx =  -1
# Swap softmax with linear
loaded_model.layers[layer_idx].activation = linear
loaded_model = utils.apply_modifications(loaded_model)


penultimate_layer_idx = utils.find_layer_idx(loaded_model, "activation_4") 
class_idx  = Ytest[0]
seed_input = Xtest[0]
grad_top1  = visualize_cam(loaded_model, layer_idx, class_idx, seed_input, 
                           penultimate_layer_idx = penultimate_layer_idx,
                           backprop_modifier     = None,
                           grad_modifier         = None)

for i in range(0,1900):
    print(i)
    class_idx  = Ytest[i]
    seed_input = Xtest[i]
    grad_top1  = visualize_cam(loaded_model, layer_idx, class_idx, seed_input, 
                               penultimate_layer_idx = penultimate_layer_idx,#None,
                               backprop_modifier     = None,
                               grad_modifier         = None)
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    axes[0].imshow((seed_input[:,:,1]* 255).astype(np.uint8), origin="lower", cmap='viridis')
    axes[1].imshow((seed_input[:,:,1]* 255).astype(np.uint8), origin="lower", cmap='viridis')
    img = axes[1].imshow(grad_top1,cmap="jet",origin="lower",alpha=0.6)
    fig.colorbar(img)
    plt.suptitle("Pr(class={},pred={}) = {:5.2f},{:5.2f},{:5.2f},{:5.2f},{:5.2f},{:5.2f},{:5.2f},{:5.2f},{:5.2f},{:5.2f},{:5.2f},{:5.2f},{:5.2f},{:5.2f},{:5.2f},{:5.2f},{:5.2f},{:5.2f},{:5.2f}".format(
                      classlabel[class_idx],classlabel[np.argmax(Ypred[i])],
                      Ypred[i][0], Ypred[i][1],Ypred[i][2],Ypred[i][3],Ypred[i][4],Ypred[i][5],Ypred[i][6],Ypred[i][7],Ypred[i][8],Ypred[i][9],Ypred[i][10],Ypred[i][11],Ypred[i][12],Ypred[i][13],Ypred[i][14],Ypred[i][15],Ypred[i][16],Ypred[i][17],Ypred[i][18]))
    plt.savefig("CAM_activation_4/CAM_TrueClass_"+str(classlabel[class_idx])+"_Pred_"+str(np.argmax(Ypred[i]))+"_index_"+str(i)+".jpg",dpi=300)
    plt.close()
