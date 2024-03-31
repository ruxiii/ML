import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from PIL import Image

# functie care citeste imaginile dintr-un director
# dat ca parametru si le pune dupa ce se normalizeaza valorile pixelilor
# intr-o lista in format de np.array
def images(images_dir, image_names, size=(32, 32)):
    images = []
    for element in image_names:
        image = Image.open(os.path.join(images_dir, element))
        images.append(np.array(image.resize(size)) / 255.0)
    return np.array(images)

# functie care citeste datele dintr-un fisier de tip csv astfel
# in functie de parametrul has_labels (daca exista si coloana 'Class') 
def csv(csv_path, column_names, has_labels=True):
    if has_labels:
        column_data = [pd.read_csv(csv_path)[column_name].tolist() for column_name in column_names]
        return column_data
    else:
        image_names = pd.read_csv(csv_path)[column_names[0]].tolist()
        return image_names
    
# citesc datele de train
train_csv_path = '/kaggle/input/unibuc-dhc-2023/train.csv'
train_images_dir = '/kaggle/input/unibuc-dhc-2023/train_images'
train_image_names, train_labels = csv(train_csv_path, ['Image', 'Class'], has_labels=True)
train_labels = to_categorical(train_labels) # transformarea etichetelor intr-un format “one-hot encoding”(vector binar cu un singur 1 si in rest 0) 
train_images = images(train_images_dir, train_image_names)

# am impartit datele de intrenament intr-un set de antrenament si unul de validare, impartind random setul de train
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.5, random_state=42)

# augumentarea datelor => imbunatatirea performantei modelului
datagen = ImageDataGenerator(
    rotation_range=20, 
    width_shift_range=0.1, 
    height_shift_range=0.1,  
    horizontal_flip=True
)

# creez layerele pentru modelul meu
model = Sequential([
#             extrag caracteristici dintr-o imagine
            Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
#             normalizez activarile stratului anterior 
            BatchNormalization(),
#             reduce dimensiunea imaginilor
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
#            transforma datele adunate dintr-o matrice intr-un vector unidimensional 
            Flatten(),
#            strat complet conectat 
            Dense(256, activation='relu'),
#            strat complet conectat, realizand distribuirea probabilitatilor pentru fiecare clasa
            Dense(96, activation='softmax')
        ])

# arhitectura modelului
model.summary()

# comiplez modelul
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
datagen.fit(train_images)

# imi voi salva cel mai bun model din punctul de vedere al acuratetii
checkpoint = tf.keras.callbacks.ModelCheckpoint('best.csv', monitor='val_accuracy', save_best_only=True, mode='max')

# dau fit
history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=32),
    epochs=100,
    validation_data=(val_images, val_labels),
    callbacks=[checkpoint]
)

# celai bun model
best = tf.keras.models.load_model('best.csv')

# citesc datele de test
test_csv_path = '/kaggle/input/unibuc-dhc-2023/test.csv'
test_images_dir = '/kaggle/input/unibuc-dhc-2023/test_images'

test_image_names = csv(test_csv_path, ['Image'], has_labels=False)
test_images = images(test_images_dir, test_image_names)

test_predictions = best.predict(test_images) #dau predict pe cel mai bun model
test_classes = np.argmax(test_predictions, axis=1)

# scriu in fisier rezultatele obtinute
test_data = {'Image': test_image_names, 'Class': test_classes}
pd.DataFrame(test_data).to_csv('test_predictions.csv', index=False)

print("CSV - created.")