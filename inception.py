import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
import cv2
import os
import pandas as pd
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths and parameters
path = "FaceShape_Dataset"
labelFile = 'FaceLabels.csv'
batch_size_val = 32
steps_per_epoch_val = 1000
epochs_val = 50  # Increased epochs for better fine-tuning
imageDimensions = (150, 150, 3)  # Increase image resolution for better feature extraction
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

            resized = cv2.resize(curImg, (150, 150))  # Resize to match InceptionV3 requirements
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
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1)])
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
X_train = preprocess_input(X_train)
X_validation = preprocess_input(X_validation)
X_test = preprocess_input(X_test)

# Data augmentation
dataGen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=20,
    horizontal_flip=True,
    fill_mode='nearest'
)
dataGen.fit(X_train)
batches = dataGen.flow(X_train, y_train, batch_size=20)
X_batch, y_batch = next(batches)

fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()
for i in range(15):
    axs[i].imshow(X_batch[i])
    axs[i].axis('off')
plt.show()

# One hot encoding of labels
y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)

# Define the model using InceptionV3 with fine-tuning
def myModel():
    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(150, 150, 3))
    
    # Unfreeze more layers of the base model for fine-tuning
    for layer in base_model.layers[-50:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(noOfClasses, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = myModel()

# Callbacks for better training
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    dataGen.flow(X_train, y_train, batch_size=batch_size_val),
    epochs=epochs_val,
    steps_per_epoch=steps_per_epoch_val,
    validation_data=(X_validation, y_validation),
    callbacks=[reduce_lr, early_stopping],
    shuffle=True
)

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
print('Test Score:', score[0])
print('Test Accuracy:', score[1])

# Save the model
model.save('faceshape_inceptionv3_finetuned_model.h5')

# Close any open CV2 windows
cv2.destroyAllWindows()
