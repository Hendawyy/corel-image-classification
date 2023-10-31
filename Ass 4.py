# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 20:45:27 2023

@author: Seif Hendawy
"""

import numpy as np 
import pandas as pd
import glob
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.decomposition import PCA
import os

for dirname, _, filenames in os.walk('CORELFeatures/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

def fe(im_grey,bSize = 8):
  fd, hog_image = hog(im_grey, orientations=9, pixels_per_cell=(bSize, bSize),
                	cells_per_block=(2, 2), visualize=True, multichannel=False)
  return fd,hog_image
im = cv2.cvtColor(cv2.imread('CORELFeatures/training_set/Architectures/264.jpg'), cv2.COLOR_BGR2GRAY)
a,b= fe(im)
a
plt.subplot(1,2,1)
plt.imshow(im,"gray")
plt.subplot(1,2,2)
plt.imshow(b,"gray")

# creating a dictionary for each class


classes_train = {'Africans':[], 'Architectures':[], 'Beach':[], 'Buses':[],'Dinosaurs':[], 'Elephants':[], 
                 'Flowers':[], 'Food':[], 'Horses':[], 'Mountain':[]}
n=0
for i,j in classes_train.items():
    for file in glob.iglob('CORELFeatures/training_set/'+i+'/*.jpg'):
        z=list(classes_train.keys())[n]  
        if(i ==z):
           print(i)
           print(j)
           print(file)
           classes_train[i]= fe(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY))
    n=n+1
            
 # reading the image and converting it into greyscale then flatten it to 1D array before adding it to its own class
df_train_wide = pd.DataFrame(classes_train)
df_train_long = pd.melt(df_train_wide,var_name="class",value_name="image") # converting the dataframe from wide format to long format
df_train_long



# redoing the steps for the test data set 
classes_test = {'Africans':[], 'Architectures':[], 'Beach':[], 'Buses':[],'Dinosaurs':[], 'Elephants':[], 
                 'Flowers':[], 'Food':[], 'Horses':[], 'Mountain':[]}
m=0
for k,l in classes_test.items():
    for filez in glob.iglob('CORELFeatures/test_set/'+k+'/*.jpg'):
        za=list(classes_test.keys())[m]  
        if(k ==za):
           print(k)
           print(l)
           print(filez)
           classes_test[k]= fe(cv2.cvtColor(cv2.imread(filez), cv2.COLOR_BGR2GRAY))
    m=m+1

df_test_wide = pd.DataFrame(classes_test)
df_test_long = pd.melt(df_test_wide,var_name="class",value_name="image")
df_test_long

# reformating the dataset such that each pixle value is a feature
y_train = df_train_long["class"].values
X_train = pd.DataFrame(df_train_long['image'].to_list()).values

y_test = df_test_long["class"].values
X_test = pd.DataFrame(df_test_long['image'].to_list()).values
X_test
X_train

X,y = (np.concatenate((X_train,X_test)),np.concatenate((y_train,y_test)))
# reducing the diminsionality of the data using PCA
pca = PCA()
pca.fit(X)
F = pca.transform(X)
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('eigen values')
plt.grid()
plt.show()

cumsum = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cumsum)
plt.title('PC variance explained')
plt.xlabel('Principal Component')
plt.ylabel('% Variance Explained')
plt.grid()
plt.show()

# reducing the diminsionality of the data using PCA
pca = PCA(n_components=59)
pca.fit(X)
F = pca.transform(X)

# normalizing data
F = F-np.mean(F)/np.std(F)



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(F,y,test_size=0.3)



from sklearn.svm import NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt



def crossValidation(k,model,X,y):
  foldsAccList = []
  foldsRecall = []
  foldsPrecesion = []
  foldsCM = []
  f1_scores = []
  kf = StratifiedKFold(k)
  rr = []
  cm = []
  for trIdx,tesIdx in kf.split(X,y):
    X_train1 = X[trIdx]
    X_test1 = X[tesIdx]
    y_train1 = y[trIdx]
    y_test1 = y[tesIdx]

    model.fit(X_train1,y_train1)
    preds = model.predict(X_test1)
    cm.append(confusion_matrix(y_true = y_test1,y_pred = preds))  
    rr.append(pd.DataFrame(classification_report(y_true=y_test1,y_pred =preds,output_dict = True)))
  return sum(rr) / k,np.sum(cm,axis=0)

#KNN with K = 5 Using The Distance Equation Minkowski.
ClassifierM = KNeighborsClassifier(n_neighbors=5, metric ='minkowski', p=2)
ClassifierM.fit(X_train, y_train)
y_pred = ClassifierM.predict(X_test)


#Confusion Matrix For K = Optimum Using The Distance Equation Minkowski.
k5cfM = confusion_matrix(y_test, y_pred)
print('Confusion Matrix : ')
print(k5cfM)

#Classification Report For K = Optimum Using The Distance Equation Minkowski.
k5crM = classification_report(y_test, y_pred)
print('Classification Report : ')
print(k5crM)

#Accuracy Score For K = Optimum Using The Distance Equation Minkowski.
k5asM = accuracy_score(y_test, y_pred)
print('Accuracy :',k5asM*100,'%')



#Multi-layer Perceptron
mlp = MLPClassifier(activation = 'relu',hidden_layer_sizes=400,solver='lbfgs')
mlp.fit(X_train,y_train)
print(classification_report(y_pred = mlp.predict(X_test),y_true = y_test))
confusion_matrix(y_pred=mlp.predict(X_test),y_true=y_test)


model =  MLPClassifier(activation = 'relu',hidden_layer_sizes=400,solver='lbfgs')
noFolds = 5
amlp,bmlp=crossValidation(noFolds,model,F,y)
print(bmlp)
amlp.T


#Support Vector Model
svmodel = NuSVC(tol = 1e-5,kernel='rbf')
svmodel.fit(X_train,y_train)
print(classification_report(y_pred = svmodel.predict(X_test),y_true = y_test))
confusion_matrix(svmodel.predict(X_test),y_test)


model =NuSVC(tol = 1e-5,kernel='rbf')
noFolds = 5
asvm,bsvm=crossValidation(noFolds,model,F,y)
print(bsvm)
asvm.T


#Random Forest
rf = RandomForestClassifier(criterion='entropy')
rf.fit(X_train,y_train)
print(classification_report(y_pred = rf.predict(X_test),y_true = y_test))
confusion_matrix(rf.predict(X_test),y_test)



model =RandomForestClassifier(criterion='entropy')
noFolds = 5
aRf,bRf=crossValidation(noFolds,model,F,y)
print(bRf)
aRf.T




#Intialize Data to be Put in The Dafarame. 
SCORESx = {'Accuracy' : pd.Series([k5asM, bmlp, bsvm, bRf], 
                                   index =['KNN', 'Multi-layer Perceptron', 
                                           'Support Vector Model', 'Random Forest'])} 

#Create an Comparison Dataframe That Holds a comparison between The Accuracy Scores and the confusion matrices of the diffrent Distance Functions. 
K5p = pd.DataFrame(SCORESx) 


axz=pd.DataFrame(K5p)


axz