# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 19:23:34 2024

@author: hp
"""
 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.datasets import fetch_openml
from skimage.feature import hog



#Load Dataset MNIST
mnist = fetch_openml("mnist_784")
images= mnist.data.astype("uint8").values
labels = mnist.target.astype("uint8")

# Split dataset antara training dan testing dataset
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)


#Ekstraksi feature HOG
def extract_hog_features(images):
    list_features = []
    for image in images:
        image_reshape = image.reshape(28, 28)
        
        features,hog_images = hog(image_reshape, orientations=8, pixels_per_cell=(4, 4),
                            cells_per_block=(1, 1), visualize=True)
        list_features.append(features)
    return np.array(list_features)

X_train_hog = extract_hog_features(X_train)
X_test_hog = extract_hog_features(X_test)

#Train processs on SVM clasifier

clf = svm.SVC()
clf.fit(X_train_hog, y_train)

#input image
input_image = y_train[1]
plt.imshow(input_image, cmap ='gray')
Label_predict = y_test[1]
#Predict
y_pred = clf.predict(X_test_hog)

#Evaluasi Performa
C_matrix = confusion_matrix(y_test, y_pred)
Accuracy = accuracy_score(y_test, y_pred)
Precision = precision_score(y_test, y_pred, average = 'weighted')

#Display
print("Confusion Matrix;\n", C_matrix)

print("Accuracy:", Accuracy)
print("Precision:", Precision)