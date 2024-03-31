import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix

# functie care citeste imaginile dintr-un director
# dat ca parametru si le pune intr-o lista in format de np.array
def images(images_dir, image_names):
    images = []
    for element in image_names:
        image = plt.imread(os.path.join(images_dir, element))
        images.append(image.flatten())
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
train_images = images(train_images_dir, train_image_names)

scaler = preprocessing.StandardScaler()  #initializez un obiect scaler care ma va ajuta sa aduc imaginile la o forma standard
scaler.fit(train_images)

train_images = scaler.transform(train_images)

# citesc val
val_csv_path = '/kaggle/input/unibuc-dhc-2023/val.csv'
val_images_dir = '/kaggle/input/unibuc-dhc-2023/val_images'
val_image_names, val_labels = csv(val_csv_path, ['Image', 'Class'], has_labels=True)
val_images = images(val_images_dir, val_image_names)

# gasirea numarul de vecini cu acuratetea cea mai buna
max_acc=-1 
n_max=-1
accuracies = []
for n in range(1,20):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(train_images, train_labels)

    predictions = knn.predict(val_images)

    accuracy = accuracy_score(val_labels, predictions)
    accuracies.append(accuracy)
    if accuracy > max_acc:
        max_acc=accuracy
        n_max=n
        print(str(n) + ":" + str(max_acc))
        
val_images = scaler.transform(val_images) #scalez imaginile
       
# creez modelul KNN cu numarul de vecini pe care l-am gasit mai sus
knn = KNeighborsClassifier(n_neighbors=n_max)
knn.fit(train_images, train_labels)

predictions = knn.predict(val_images) #dau predict sa vad cat de bine a invatat modelul sa clasifice

accuracy = accuracy_score(val_labels, predictions) #calculez acuratetea
print(accuracy)
print(confusion_matrix(val_labels, predictions)) #calculez matricea de confuzie

# citesc datele de test
test_csv_path = '/kaggle/input/unibuc-dhc-2023/test.csv'
test_images_dir = '/kaggle/input/unibuc-dhc-2023/test_images'
test_image_names = csv(test_csv_path, ['Image', 'Class'], has_labels=False)
test_images = images(test_images_dir, test_image_names)

test_images = scaler.transform(test_images) #le standardizez
test_predictions = knn.predict(test_images) #dau predict

# scriu in fisier rezultatele obtinute
test_data = {'Image': test_image_names, 'Class': test_predictions}
pd.DataFrame(test_data).to_csv('test_predictions_6.csv', index=False)

print("CSV - created.")