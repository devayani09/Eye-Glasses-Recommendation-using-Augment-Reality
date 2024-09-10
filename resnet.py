import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import cv2
import os
import pandas as pd
import random

# Assuming previous code for data loading, preprocessing and splitting has already run

# Preprocess data specific to ResNet50 input
X_train = np.repeat(X_train, 3, axis=-1)
X_validation = np.repeat(X_validation, 3, axis=-1)
X_test = np.repeat(X_test, 3, axis=-1)

input_shape = (32, 32, 3)
resnet_input = Input(shape=input_shape)
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=resnet_input)

x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(noOfClasses, activation='softmax')(x)

model_resnet = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model_resnet.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

history_resnet = model_resnet.fit(dataGen.flow(X_train, y_train, batch_size=batch_size_val),
                                  steps_per_epoch=steps_per_epoch_val, epochs=epochs_val,
                                  validation_data=(X_validation, y_validation), shuffle=True,
                                  callbacks=[early_stopping, reduce_lr])

# Evaluate the model
score_resnet = model_resnet.evaluate(X_test, y_test, verbose=0)
print('ResNet50 Test Accuracy:', score_resnet[1])

# Save the model
model_resnet.save('faceshape_resnet50_model.h5')
