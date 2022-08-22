""" The V4 model of our test

This version we are trying to use 
wider NN for prediction density and mean size

"""

# import the necessary packages
from keras.models import Sequential
from keras.applications.resnet_v2 import ResNet50V2
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model

def create_cnn(width, height, depth, filters=(256, 128, 128), regress=False):
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	inputShape = (height, width, depth)
	chanDim = -1

	# define the model input
	inputs = Input(shape=inputShape)

	# loop over the number of filters
	for (i, f) in enumerate(filters):
		# if this is the first CONV layer then set the input
		# appropriately
		if i == 0:
			x = inputs

		# CONV => RELU => BN => POOL
		x = Conv2D(f, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)

	# flatten the volume, then FC => RELU => BN => DROPOUT
	x = Flatten()(x)
	# FC-1
	x = Dense(128)(x)
	x = Activation("relu")(x)
	x = BatchNormalization(axis=chanDim)(x)
	x = Dropout(0.5)(x)

	# FC-2
	x = Dense(128)(x)
	x = Activation("relu")(x)
	x = BatchNormalization(axis=chanDim)(x)
	x = Dropout(0.5)(x)

	# FC-3
	x = Dense(64)(x)
	x = Activation("relu")(x)
	x = BatchNormalization(axis=chanDim)(x)

	# FC-4
	x = Dense(64)(x)
	x = Activation("relu")(x)

	# check to see if the regression node should be added
	if regress:
		x = Dense(1, activation="sigmoid")(x)

	# construct the CNN
	model = Model(inputs, x)

	# return the CNN
	return model


def create_model(width, height, depth):
	# initialize the input shape and channel dimension, assuming
	# # TensorFlow/channels-last ordering
	inputShape = (height, width, depth)
	chanDim = -1
	# define the model input
	inputs = Input(shape=inputShape)
	base = ResNet50V2(include_top=False, weights=None, input_tensor=inputs, input_shape=inputShape, pooling='max')
	print(base.output)
	#x = Flatten()(base.output)
	# FC-1
	x = Dense(128)(base.output)
	x = Activation("relu")(x)
	x = BatchNormalization(axis=chanDim)(x)
	x = Dropout(0.5)(x)
	# FC-2
	x = Dense(128)(x)
	x = Activation("relu")(x)
	x = BatchNormalization(axis=chanDim)(x)
	x = Dropout(0.5)(x)
	x = Dense(1 , activation="sigmoid")(x)
	#model = Model(inputs=base.inputs, outputs=x)
	model = Model(inputs, x)
	return model