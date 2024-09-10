import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, Add, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2
import os
import pandas as pd
import random
from sklearn.metrics import r2_score
import tensorflow as tf
# Define paths and parameters
path = "FaceShape_Dataset"
labelFile = 'FaceLabels.csv'
batch_size_val = 50
steps_per_epoch_val = 1000
epochs_val = 30
imageDimensions = (32, 32, 3)
testRatio = 0.2
validationRatio = 0.2

# Initialize data containers
images = []
classNo = []
myList = os.listdir(path)
print("Total Classes Detected:", len(myList))
noOfClasses = len(myList)
print("Importing Classes.....")

if not myList:
    raise ValueError(f"No classes found in the directory {path}")

for count, folder in enumerate(myList):
    folder_path = os.path.join(path, folder)
    if not os.path.isdir(folder_path):
        print(f"Skipping non-directory: {folder_path}")
        continue

    myPicList = os.listdir(folder_path)
    if not myPicList:
        print(f"No images found in directory: {folder_path}")
        continue

    for y in myPicList:
        try:
            curImg = cv2.imread(os.path.join(folder_path, y))
            if curImg is None:
                print(f"Failed to read image: {os.path.join(folder_path, y)}")
                continue

            resized = cv2.resize(curImg, (32, 32))
            images.append(resized)
            classNo.append(count)
        except Exception as e:
            print(f"Error reading image: {e}")
    print(f"Loaded class {count}: {folder}")

images = np.array(images)
classNo = np.array(classNo)

if images.size == 0 or classNo.size == 0:
    raise ValueError("No images were loaded. Check the dataset path and image files.")

X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio, stratify=classNo)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio, stratify=y_train)

print("Data Shapes")
print("Train", X_train.shape, y_train.shape)
print("Validation", X_validation.shape, y_validation.shape)
print("Test", X_test.shape, y_test.shape)

data = pd.read_csv(labelFile)
print("Data shape", data.shape, type(data))

num_of_samples = []
cols = 5
num_classes = noOfClasses
fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 300))
fig.tight_layout()
for i in range(cols):
    for j, row in data.iterrows():
        x_selected = X_train[y_train == j]
        if len(x_selected) == 0:
            continue
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1)], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j) + "-" + row["Name"])
            num_of_samples.append(len(x_selected))

print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

X_train = np.array(list(map(preprocessing, X_train)))
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)
batches = dataGen.flow(X_train, y_train, batch_size=20)
X_batch, y_batch = next(batches)

fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()
for i in range(15):
    axs[i].imshow(X_batch[i].reshape(imageDimensions[0], imageDimensions[1]), cmap=plt.get_cmap("gray"))
    axs[i].axis('off')
plt.show()

y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)

def basicCNNModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
    
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def deeperCNNModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 1), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
    
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def residualBlock(x, filters):
    res = x
    x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, (3, 3), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = Add()([x, res])
    x = Activation('relu')(x)
    return x

def residualCNNModel():
    inputs = Input(shape=(32, 32, 1))
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = residualBlock(x, 32)
    
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = residualBlock(x, 64)
    
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)
    
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(noOfClasses, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)

basic_model = basicCNNModel()
history_basic = basic_model.fit(dataGen.flow(X_train, y_train, batch_size=batch_size_val),
                                steps_per_epoch=steps_per_epoch_val, epochs=epochs_val,
                                validation_data=(X_validation, y_validation), shuffle=True,
                                callbacks=[early_stopping, reduce_lr])
basic_score = basic_model.evaluate(X_test, y_test, verbose=0)
print('Basic CNN Model Test Accuracy:', basic_score[1])
basic_model.save('basic_faceshape_model.h5')

deeper_model = deeperCNNModel()
history_deeper = deeper_model.fit(dataGen.flow(X_train, y_train, batch_size=batch_size_val),
                                  steps_per_epoch=steps_per_epoch_val, epochs=epochs_val,
                                  validation_data=(X_validation, y_validation), shuffle=True,
                                  callbacks=[early_stopping, reduce_lr])
deeper_score = deeper_model.evaluate(X_test, y_test, verbose=0)
print('Deeper CNN Model Test Accuracy:', deeper_score[1])
deeper_model.save('deeper_faceshape_model.h5')

residual_model = residualCNNModel()
history_residual = residual_model.fit(dataGen.flow(X_train, y_train, batch_size=batch_size_val),
                                      steps_per_epoch=steps_per_epoch_val, epochs=epochs_val,
                                      validation_data=(X_validation, y_validation), shuffle=True,
                                      callbacks=[early_stopping, reduce_lr])
residual_score = residual_model.evaluate(X_test, y_test, verbose=0)
print('Residual CNN Model Test Accuracy:', residual_score[1])
residual_model.save('residual_faceshape_model.h5')

print("Basic CNN Model Test Accuracy:", basic_score[1])
print("Deeper CNN Model Test Accuracy:", deeper_score[1])
print("Residual CNN Model Test Accuracy:", residual_score[1])

def plot_history(history, title):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{title} - Loss')
    plt.legend(['train', 'validation'])
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{title} - Accuracy')
    plt.legend(['train', 'validation'])
    plt.show()

plot_history(history_basic, 'Basic CNN Model')
plot_history(history_deeper, 'Deeper CNN Model')
plot_history(history_residual, 'Residual CNN Model')
