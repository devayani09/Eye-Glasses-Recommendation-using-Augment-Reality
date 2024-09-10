import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2
import os
import pandas as pd
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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

# Check if the directory exists and is not empty
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

# Convert lists to numpy arrays
images = np.array(images)
classNo = np.array(classNo)

# Check if the dataset is not empty
if images.size == 0 or classNo.size == 0:
    raise ValueError("No images were loaded. Check the dataset path and image files.")

# Split data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio, stratify=classNo)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio, stratify=y_train)

# Print data shapes
print("Data Shapes")
print("Train", X_train.shape, y_train.shape)
print("Validation", X_validation.shape, y_validation.shape)
print("Test", X_test.shape, y_test.shape)

# Load and display label data
data = pd.read_csv(labelFile)
print("Data shape", data.shape, type(data))

# Display sample images
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

# Preprocess images
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

# Data augmentation
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10,
                             horizontal_flip=True)
dataGen.fit(X_train)
batches = dataGen.flow(X_train, y_train, batch_size=20)
X_batch, y_batch = next(batches)

fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()
for i in range(15):
    axs[i].imshow(X_batch[i].reshape(imageDimensions[0], imageDimensions[1]), cmap=plt.get_cmap("gray"))
    axs[i].axis('off')
plt.show()

# One hot encoding of labels
y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)

# Define the model
def myModel():
    no_Of_Filters = 60
    size_of_Filter = (5, 5)
    size_of_Filter2 = (3, 3)
    size_of_pool = (2, 2)
    no_Of_Nodes = 500

    model = Sequential()
    model.add(Conv2D(no_Of_Filters, size_of_Filter, input_shape=(imageDimensions[0], imageDimensions[1], 1), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(no_Of_Filters, size_of_Filter, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=size_of_pool))
    
    model.add(Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(no_Of_Nodes, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
    
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = myModel()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

history = model.fit(X_train, y_train, epochs=epochs_val, steps_per_epoch=steps_per_epoch_val, batch_size=batch_size_val,
                    validation_data=(X_validation, y_validation), shuffle=True,
                    callbacks=[early_stopping, reduce_lr])

print(model.summary())

# Plot training history
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.show()

# Evaluate the model
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy:', score[1])

# Save the model
model.save('faceshape_model.h5')

# Close any open CV2 windows
cv2.destroyAllWindows()
