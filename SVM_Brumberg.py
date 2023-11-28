#!/usr/bin/env python
# coding: utf-8

# In[1]:


#SUPPORT VECTOR MACHINES

#Imports

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import random as rd
from sklearn.metrics import confusion_matrix


#!pip install pydotplus
import pydotplus

#!pip install graphviz
#!conda install -c anaconda graphviz
import graphviz 


# In[2]:


#Data preparation

Bahia = pd.read_csv('BahiaSubsample_full.csv')

#Remove duplicate rows. For duplicates, remove the row version with fewest NAs
Bahia['nan_count'] = Bahia.isna().sum(axis=1)
Bahia = Bahia.sort_values(by=['CD_MUN', 'nan_count']).drop_duplicates(subset='CD_MUN', keep='first')
Bahia = Bahia.drop(columns=['nan_count'])


#Create labels by sorting data into three classes by the Percent Area Restored 

# Get the quantile values
q1 = Bahia['PercRest'].quantile(0.33)
q2 = Bahia['PercRest'].quantile(0.66)

# Create classes based on the quantiles
bins = [-float('inf'), q1, q2, float('inf')]
labels = ['LowRest', 'MidRest', 'HighRest']

Bahia['Label'] = pd.cut(Bahia['PercRest'], bins=bins, labels=labels, include_lowest=True)


# Count the number of munipalities in each class
label_counts = Bahia['Label'].value_counts()
print(label_counts)

Bahia.to_csv('BahiaSubsample_labeled.csv', index=False)


# In[3]:


# Create box plot to show the distribution of points in three classes
plt.figure(figsize=(10, 6))
sns.boxplot(x='Label', y='PercRest', data=Bahia, order=['LowRest', 'MidRest', 'HighRest'])

plt.title('Distribution of Percent Area Restored Across Classes')
plt.ylabel('Percent area restored')
plt.xlabel('Class')
plt.grid(axis='y')

plt.show()


# In[4]:


#Remove columns not using in DT, including columns with NAs

#Removing these columns because not of interest in SVM analysis or categorical
cols_to_drop = [0, 1, 2, 3, 7, 8, 9, 14, 15, 16, 18]
drop_column_names = Bahia.columns[cols_to_drop]
Bahia = Bahia.drop(columns=drop_column_names)


# Also remove columns with NaN values because SVM in sklearn won't work with NAn=N values
Bahia = Bahia.dropna(axis=1, how='any')
#Unfortunately this means that we lose data on tree extraction value and crop value because there are NAs in these columns.
#These NAs likely represent data not being complete, not values of 0, because there is crop land in these areas 
#(so crop value is almost definitely not 0, and tree extraction value is unclear)
#Aother approach would be to remove rows with NA values in these columns. 
#However, for this sample dataset, we have so few observations (n=164), that we are trying to avoid losing observations
#Future analysis with more municipalities could retain columns with NAs and instead remove observations with NA values

# Display the first few rows to verify
print(Bahia.head())

Bahia.to_csv('BahiaSubsample_SVM.csv', index=False)


# In[5]:


#Create testing and training sets
rd.seed(1234)
TrainDF, TestDF = train_test_split(Bahia, test_size=0.3)

print(f"Number of rows in training dataset: {len(TrainDF)}")
print(f"Number of rows in testing dataset: {len(TestDF)}")

#TrainDF.to_csv('TrainDF.csv', index=False)
#TestDF.to_csv('TestDF.csv', index=False)

#Need to remove labels for DT

#Save test labels as separate DF
TestLabels=TestDF["Label"]
#print(TestLabels)

#Remove labels
TestDF = TestDF.drop(["Label"], axis=1)
#print(TestDF)

#Save train labels as separate DF
TrainLabels=TrainDF["Label"]
#print(TrainLabels)

# Remove labels
TrainDF = TrainDF.drop(["Label"], axis=1)
#print(TrainDF)


# In[6]:


# Scale all data between 0 and 1

#TrainDF
x = TrainDF.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
TrainDF = pd.DataFrame(x_scaled, columns=TrainDF.columns, index=TrainDF.index)

# TestDF using same scaler as TrainDF
x2 = TestDF.values
x_scaled2 = min_max_scaler.transform(x2)  # Use transform, not fit_transform
TestDF = pd.DataFrame(x_scaled2, columns=TestDF.columns, index=TestDF.index)

#print(TestDF)


# In[7]:


#SUPPORT VECTOR MACHINES (SVM)

from sklearn.svm import LinearSVC

SVM_Model1=LinearSVC(C=1)
SVM_Model1.fit(TrainDF, TrainLabels)

#print("SVM 1 prediction:\n", SVM_Model1.predict(TestDF))
#print("Actual:")
#print(TestLabels)

SVM_matrix = confusion_matrix(TestLabels, SVM_Model1.predict(TestDF))
print("\nThe confusion matrix for Linear SVM is:")
print(SVM_matrix)
print("\n\n")


# See order of classes
class_labels = SVM_Model1.classes_
print("Class labels:", class_labels)
# ['HighRest' 'LowRest' 'MidRest']



# In[8]:


#Feature importance of linear SVM 

SVM_Model = LinearSVC(C=1)
SVM_Model.fit(TrainDF, TrainLabels)
coefficients = SVM_Model.coef_[0]
feature_names = TrainDF.columns if hasattr(TrainDF, 'columns') else range(len(coefficients))

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(feature_names, coefficients)
plt.xticks(rotation=45)
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.title('Feature Importance in Linear SVM, C=1')
plt.show()


# In[9]:


#SVM with other kernels

#Radial Basis Function (RBF) kernel 
SVM_Model2=sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma="auto")
SVM_Model2.fit(TrainDF, TrainLabels)

#print("SVM prediction:\n", SVM_Model2.predict(TestDF))
#print("Actual:")
#print(TestLabels)

SVM_matrix = confusion_matrix(TestLabels, SVM_Model2.predict(TestDF))
print("\nThe confusion matrix for RBF SVM is:")
print(SVM_matrix)
print("\n\n")


# In[10]:


# Polynomial kernel function 
SVM_Model3 = sklearn.svm.SVC(C=1.0, kernel='poly', degree=3, gamma="auto")
#SVM_Model3=sklearn.svm.SVC(C=1.0, kernel='poly', degree=3, gamma="scale")
SVM_Model3.fit(TrainDF, TrainLabels)

#print("SVM prediction:\n", SVM_Model3.predict(TestDF))
#print("Actual:")
#print(TestLabels)

SVM_matrix = confusion_matrix(TestLabels, SVM_Model3.predict(TestDF))
print("\nThe confusion matrix for poly p = 3 SVM is:")
print(SVM_matrix)
print("\n\n")


# In[11]:


#Compare Linear SVM with different C values

# Select C values to compare
C_values = [0.1, 1, 10]

models = []
predictions = []
conf_matrices = []

for C in C_values:
    model = LinearSVC(C=C)
    model.fit(TrainDF, TrainLabels)
    pred = model.predict(TestDF)
    conf_matrix = confusion_matrix(TestLabels, pred)
    
    models.append(model)
    predictions.append(pred)
    conf_matrices.append(conf_matrix)

    print(f"SVM with C={C} prediction:\n", pred)
    print("Actual:")
    print(TestLabels)
    print("\nThe confusion matrix for Linear SVM with C={C} is:")
    print(conf_matrix)
    print("\n\n")


#Make figure to compare confusion matrices and accuracy percentages
    
from sklearn.metrics import accuracy_score

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Confusion Matrices for Different C values in Linear SVM')

for i, (C, conf_matrix) in enumerate(zip(C_values, conf_matrices)):
    accuracy = accuracy_score(TestLabels, predictions[i])
    title = f'Linear SVM with C={C} (Accuracy: {accuracy:.2f}%)'
    sns.heatmap(conf_matrix, annot=True, fmt="d", ax=axes[i], cmap="Blues")
    axes[i].set_title(title)
    axes[i].set_xlabel('Predicted labels')
    if i == 0:
        axes[i].set_ylabel('True labels')

plt.tight_layout()
plt.subplots_adjust(top=0.85)  
plt.show()


# In[12]:


#Compare RBF SVM with different C values

from sklearn.svm import SVC

# Select C values for RBF to compare
RBF_C_values = [0.1, 1, 10]

# Store RBF models, predictions, and confusion matrices
RBF_models = []
RBF_predictions = []
RBF_conf_matrices = []

for C in RBF_C_values:
    RBF_model = SVC(C=C, kernel='rbf', gamma="auto")
    RBF_model.fit(TrainDF, TrainLabels)
    RBF_pred = RBF_model.predict(TestDF)
    RBF_conf_matrix = confusion_matrix(TestLabels, RBF_pred)
    
    RBF_models.append(RBF_model)
    RBF_predictions.append(RBF_pred)
    RBF_conf_matrices.append(RBF_conf_matrix)

    print(f"RBF SVM with C={C} prediction:\n", RBF_pred)
    print("Actual:")
    print(TestLabels)
    print(f"\nThe confusion matrix for RBF SVM with C={C} is:")
    print(RBF_conf_matrix)
    print("\n\n")
    
    
#Make figure to compare RBF confusion matrices and accuracy percentages

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Confusion Matrices for Different C values in RBF SVM')

for i, (C, RBF_conf_matrix) in enumerate(zip(RBF_C_values, RBF_conf_matrices)):
    RBF_accuracy = accuracy_score(TestLabels, RBF_predictions[i])
    RBF_title = f'RBF SVM with C={C} (Accuracy: {RBF_accuracy:.2f}%)'
    sns.heatmap(RBF_conf_matrix, annot=True, fmt="d", ax=axes[i], cmap="Blues")
    axes[i].set_title(RBF_title)
    axes[i].set_xlabel('Predicted labels')
    if i == 0:
        axes[i].set_ylabel('True labels')

plt.tight_layout()
plt.subplots_adjust(top=0.85)  
plt.show()


# In[13]:


#Polynomial SVM models with different degrees and C values

# degrees and C values want to compare
poly_degrees = [2, 3, 4]
C_values = [0.1, 1, 10]

poly_models = {}
poly_predictions = {}
poly_conf_matrices = {}

for degree in poly_degrees:
    for C in C_values:
        model_key = (degree, C)
        poly_model = SVC(C=C, kernel='poly', degree=degree, gamma="auto")
        poly_model.fit(TrainDF, TrainLabels)
        poly_pred = poly_model.predict(TestDF)
        poly_conf_matrix = confusion_matrix(TestLabels, poly_pred)

        poly_models[model_key] = poly_model
        poly_predictions[model_key] = poly_pred
        poly_conf_matrices[model_key] = poly_conf_matrix

        print(f"Polynomial SVM with degree={degree} and C={C} prediction:\n", poly_pred)
        print("Actual:")
        print(TestLabels)
        print(f"\nThe confusion matrix for Polynomial SVM with degree={degree} and C={C} is:")
        print(poly_conf_matrix)
        print("\n\n")

        
#Make figure to compare confusion matrices and accuracy percentages for 3 degrees and 3 C values

fig, axes = plt.subplots(len(poly_degrees), len(C_values), figsize=(15, 15))
fig.suptitle('Confusion Matrices for Different Degrees and C values in Polynomial SVM')

for i, degree in enumerate(poly_degrees):
    for j, C in enumerate(C_values):
        model_key = (degree, C)
        poly_accuracy = accuracy_score(TestLabels, poly_predictions[model_key])
        poly_title = f'Degree={degree}, C={C}\nAccuracy: {poly_accuracy:.2f}%'
        ax = axes[i, j]  
        sns.heatmap(poly_conf_matrices[model_key], annot=True, fmt="d", ax=ax, cmap="Blues")
        ax.set_title(poly_title)
        if i == len(poly_degrees) - 1:
            ax.set_xlabel('Predicted labels')
        if j == 0:
            ax.set_ylabel('True labels')

plt.tight_layout()
plt.subplots_adjust(top=0.9) #moving the title up because it overlapped with the first row of figures
plt.show()

