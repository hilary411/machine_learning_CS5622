#DECISION TREES & NAIVE BAYES
#HILARY BRUMBERG

# In[1]:

#Imports

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import random as rd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
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

Bahia.to_csv('BahiaSubsample_full.csv', index=False)


# In[3]:


# Create box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Label', y='PercRest', data=Bahia, order=['LowRest', 'MidRest', 'HighRest'])

plt.title('Distribution of Percent Area Restored Across Classes')
plt.ylabel('Percent area restored')
plt.xlabel('Class')
plt.grid(axis='y')

plt.show()


# In[4]:


#Remove columns not using in DT, including columns with NAs

#Removing these columsn because not of interest in DT analysis
cols_to_drop = [0, 1, 2, 3, 7, 8, 9, 14, 15, 16, 18]
drop_column_names = Bahia.columns[cols_to_drop]
Bahia = Bahia.drop(columns=drop_column_names)

# Also remove columns with NaN values because DT in sklearn won't work with NAn=N values
Bahia = Bahia.dropna(axis=1, how='any')
#Unfortunately this means that we lose data on tree extraction value and crop value because there are NAs in these columns.
#These NAs likely represent data not being complete, not values of 0, because there is crop land in these areas 
#(so crop value is almost definitely not 0, and tree extraction value is unclear)
#Aother approach would be to remove rows with NA values in these columns. 
#However, for this sample dataset, we have so few observations (n=164), that we are trying to avoid losing observations
#Future analysis with more municipalities could retain columns with NAs and instead remove observations with NA values

# Display the first few rows to verify
print(Bahia.head())

Bahia.to_csv('BahiaSubsample_DT.csv', index=False)


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
TrainDF_S = pd.DataFrame(x_scaled, columns=TrainDF.columns, index=TrainDF.index)

# TestDF using same scaler as TrainDF
x2 = TestDF.values
x_scaled2 = min_max_scaler.transform(x2)  # Use transform, not fit_transform
TestDF_S = pd.DataFrame(x_scaled2, columns=TestDF.columns, index=TestDF.index)

#print(TestDF_S)


# In[7]:


#Decision tree

MyDT_R=DecisionTreeClassifier(criterion='entropy', ##"entropy" or "gini"
                            splitter='best',  ## or "random" or "best"
                            max_depth=None, 
                            min_samples_split=2, 
                            min_samples_leaf=1, 
                            min_weight_fraction_leaf=0.0, 
                            max_features=None, 
                            random_state=None, 
                            max_leaf_nodes=None, 
                            min_impurity_decrease=0.0, 
                            #min_impurity_split=None, 
                            class_weight=None)

## perform DT
MyDT_R.fit(TrainDF_S, TrainLabels)
    ## plot the tree
tree.plot_tree(MyDT_R)

feature_namesR=TrainDF_S.columns
print(feature_namesR)


# In[8]:


#Visualize results

TREE_data = tree.export_graphviz(MyDT_R, out_file=None,
                  feature_names=TrainDF_S.columns,
                  filled=True, 
                  rounded=True,  
                  special_characters=True) 
                                   
graph = graphviz.Source(TREE_data) 
graph.render("Tree_Record") 


# In[9]:


#Confusion matrix

#determine order of classes
class_labels = MyDT_R.classes_
print(class_labels)

#Show the predictions from the DT on the test set

DT_pred_R=MyDT_R.predict(TestDF_S)
bn_matrix_R = confusion_matrix(TestLabels, DT_pred_R)
print("\nThe confusion matrix is:")
print(bn_matrix_R)


# In[10]:


## Feature Importance
FeatureImpR=MyDT_R.feature_importances_   
indicesR = np.argsort(FeatureImpR)[::-1]
indicesR
print ("feature name: ", feature_namesR[indicesR])

## print out the important features.....
for f in range(TrainDF_S.shape[1]):
    if FeatureImpR[indicesR[f]] > 0:
        print("%d. feature %d (%f)" % (f + 1, indicesR[f], FeatureImpR[indicesR[f]]))
        print ("feature name: ", feature_namesR[indicesR[f]])

#Interpretation:
#For the first Decision Tree generated:
#Farming_perc_2011 is the most influential feature. 
#GDP per capita is second
#Population is the third most influential feature
#Natural vegetation total percent area in 2011 is also similarly important
#Animal Production and cattle heads are lower importance
#Forest perct area is also of lower importance


# In[11]:


##  Visualize Decision Trees plotting paired surfaces

from sklearn.tree import DecisionTreeClassifier, plot_tree

f1=TrainDF_S.columns.get_loc("Farming_perc_2011") 
f2=TrainDF_S.columns.get_loc("GDPperCap") 

n_classes =2
plot_colors = "ryb"
plot_step = 0.02

for pairidx, pair in enumerate([[f1, f2], [0, 2], [0, 3],
                                [1, 2], [1, 3]]):
    #print(TrainDF1.iloc[:,pair])
    X = TrainDF_S.iloc[:, pair]
    ## Because we are plotting, using our GOD and HIKE labels will not work
    ## we need to change them to 0 and 1
    y = TrainLabels
    #print(y)
    oldy=y
    #print(type(y))
    y=y.replace('LowRest', 1)
    y=y.replace('HighRest', 0)
    y=y.replace('MidRest', 2)
    
    #print(y)
    # Train
    DTC = DecisionTreeClassifier().fit(X, y)
    # Plot the decision boundary
    plt.subplot(2, 3, pairidx + 1)

    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    print(x_min)
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
   
    xx, yy = np.meshgrid(np.arange(x_min, x_max,plot_step),
                         np.arange(y_min, y_max,plot_step))
    
    #print(yy)
    
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
#
    Z = DTC.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    #print(Z)
    
    
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
       
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], s=30, label=oldy,edgecolor='black', 
                    #c=color, s=15)
                    #label=y[i],
                    cmap=plt.cm.RdYlBu)
###---------------------------end for loop ----------------------------------
plt.suptitle("Decision surface of a decision tree using paired features: Farming area before restoration vs. GDP per capita")
#plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")
plt.figure()


# In[12]:


#Another way to visualize the decision tree

import six
import sys
sys.modules['sklearn.externals.six'] = six
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz

dot_data2 = StringIO()

export_graphviz(MyDT_R, out_file=dot_data2,  
                 filled=True, rounded=True,
                 special_characters=True,
                 feature_names = TrainDF.columns,
                 class_names=['HighRest', 'LowRest', 'MidRest'])
graph = pydotplus.graph_from_dot_data(dot_data2.getvalue())  
graph.write_png('DecisionTree.png')


# In[13]:


#RandomForest
from sklearn.ensemble import RandomForestClassifier


RF1 = RandomForestClassifier()
RF1.fit(TrainDF_S, TrainLabels)
RF1_pred=RF1.predict(TestDF_S)

bn_matrix_RF = confusion_matrix(TestLabels, RF1_pred)
print("\nThe confusion matrix is:")
print(bn_matrix_RF)

#Visualize random forest
Features=TrainDF_S.columns
#Targets=TestLabels

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(RF1.estimators_[0],
               feature_names = Features, 
               #class_names=Targets,
               filled = True)

fig.savefig('RF_Tree')

#View estimator Trees in RF

fig2, axes2 = plt.subplots(nrows = 1,ncols = 3,figsize = (10,2), dpi=900)
for index in range(0, 3):
    tree.plot_tree(RF1.estimators_[index],
                   feature_names = Features, 
                   filled = True,
                   ax = axes2[index])

    axes2[index].set_title('Estimator: ' + str(index), fontsize = 11)
fig2.savefig('THREEtrees_RF.png')


# In[14]:


#Decision tree WIHTOUT scaling parameters 0-1

MyDT_R2=DecisionTreeClassifier(criterion='entropy', ##"entropy" or "gini"
                            splitter='best',  ## or "random" or "best"
                            max_depth=None, 
                            min_samples_split=2, 
                            min_samples_leaf=1, 
                            min_weight_fraction_leaf=0.0, 
                            max_features=None, 
                            random_state=None, 
                            max_leaf_nodes=None, 
                            min_impurity_decrease=0.0, 
                            #min_impurity_split=None, 
                            class_weight=None)

## perform DT
MyDT_R2.fit(TrainDF, TrainLabels)
    ## plot the tree
tree.plot_tree(MyDT_R2)

feature_namesR=TrainDF.columns
print(feature_namesR)

TREE_data2 = tree.export_graphviz(MyDT_R2, out_file=None,
                  feature_names=TrainDF.columns,
                  filled=True, 
                  rounded=True,  
                  special_characters=True) 
                                   
graph = graphviz.Source(TREE_data2) 
graph.render("Tree_Record4") 

#determine order of classes
class_labels = MyDT_R2.classes_
print(class_labels)

#Show the predictions from the DT on the test set

DT_pred_R2=MyDT_R2.predict(TestDF)
bn_matrix_R = confusion_matrix(TestLabels, DT_pred_R2)
print("\nThe confusion matrix is:")
print(bn_matrix_R)


# In[15]:


#Naive Bayes

from sklearn.naive_bayes import MultinomialNB
MyModelNB_Num= MultinomialNB()

MyModelNB_Num.fit(TrainDF_S, TrainLabels)
PredictionNB = MyModelNB_Num.predict(TestDF_S)

#Confusion matrix
from sklearn.metrics import confusion_matrix

#Our confusion matrix is 3X3
#Rows are the true labels
#Columns are predicted
#It is alphabetical, so High, Low, and then Mid

cnf_matrix = confusion_matrix(TestLabels, PredictionNB)
print("\nThe confusion matrix is:")
print(cnf_matrix)



# In[16]:


#Prediction probabilities
#columns are the labels in alphabetical order
#The decimal in the matrix are the prob of being that label

print(np.round(MyModelNB_Num.predict_proba(TestDF_S),2))
MyModelNB_Num.get_params(deep=True)

#Visualize the prediction probabilities
probs = np.round(MyModelNB_Num.predict_proba(TestDF_S), 2)

barWidth = 0.25
r1 = np.arange(len(probs))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

plt.figure(figsize=(15,7))
plt.bar(r1, probs[:, 0], width=barWidth, edgecolor='white', label='HighRest')
plt.bar(r2, probs[:, 1], width=barWidth, edgecolor='white', label='LowRest')
plt.bar(r3, probs[:, 2], width=barWidth, edgecolor='white', label='MidRest')

plt.xlabel('Instances', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(probs))], range(1,len(probs)+1))  # Numbering instances for clarity
plt.ylabel('Probability', fontweight='bold')
plt.title('Prediction Probabilities for Each Class')

plt.legend()
plt.tight_layout()
plt.show()


#Line graph

# Extract probabilities for each class
highRest_probs = probs[:, 0]
lowRest_probs = probs[:, 1]
midRest_probs = probs[:, 2]

data = [highRest_probs, lowRest_probs, midRest_probs]

# Plotting
fig, ax = plt.subplots(figsize=(10,7))

# Creating boxplot
ax.boxplot(data, vert=True, patch_artist=True, labels=['HighRest', 'LowRest', 'MidRest'])

# Labeling and other aesthetics
ax.set_title('Boxplot of Prediction Probabilities for Each Class')
ax.set_xlabel('Class')
ax.set_ylabel('Probability')
ax.grid(True, which='both', linestyle='--', linewidth=0.5, axis='y', alpha=0.7)

plt.show()


# In[17]:


#Visualize Naive Bayes with PCA (2 principal components)

from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt

## remap labels to numbers to view
ymap=TrainLabels
ymap=ymap.replace("LowRest", 1)
ymap=ymap.replace("HighRest", 0)
ymap=ymap.replace("MidRest", 2)

pca = PCA(n_components=2)
proj = pca.fit_transform(TrainDF_S)

fig, ax = plt.subplots()
colors = ["r", "g", "b"]
labels = ["HighRest", "LowRest", "MidRest"]
for i, color, label in zip([0, 1, 2], colors, labels):
    ax.scatter(proj[ymap == i, 0], proj[ymap == i, 1], c=color, label=label)

ax.legend()
ax.set_xlabel("First Principal Component")
ax.set_ylabel("Second Principal Component")

plt.show()


# In[18]:


#3 principle components


pca = PCA(n_components=3)
proj = pca.fit_transform(TrainDF_S)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ["r", "g", "b"]
labels = ["HighRest", "LowRest", "MidRest"]

for i, color, label in zip([0, 1, 2], colors, labels):
    ax.scatter(proj[ymap == i, 0], proj[ymap == i, 1], proj[ymap == i, 2], c=color, label=label)

ax.legend()
plt.show()

