
# In[2]:


#Amazon data analysis for Machine Learning exam 2
#Hilary Brumberg

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
import graphviz 

# In[3]:


#Data preparation

AmazonDF = pd.read_csv('MLFinalExamData.csv')

#Create dataframes for decision tree and association rule mining

#Removing this column because not of interest in DT analysis
drop_DT = [0]
names_DT = AmazonDF.columns[drop_DT]
AmazonDT = AmazonDF.drop(columns=names_DT)
#print(AmazonDT)


#Removing these columns because not of interest in ARM analysis
drop_ARM = [1,2,3]
names_ARM = AmazonDF.columns[drop_ARM]
AmazonARM = AmazonDF.drop(columns=names_ARM)
#print(AmazonARM)


# In[4]:


#Problem 1: determine if a User should get a credit card (yes or no)
#Method: Decision tree

#Create testing and training sets
rd.seed(1234)
TrainDF, TestDF = train_test_split(AmazonDT, test_size=0.3)

print(f"Number of rows in training dataset: {len(TrainDF)}")
print(f"Number of rows in testing dataset: {len(TestDF)}")

#Need to remove labels for DT
#Labels are "Credit_card"

#Save test labels as separate DF
TestLabels=TestDF["Credit_card"]
print(TestLabels)

#Remove labels
TestDF = TestDF.drop(["Credit_card"], axis=1)
#print(TestDF)

#Save train labels as separate DF
TrainLabels=TrainDF["Credit_card"]
#print(TrainLabels)

# Remove labels
TrainDF = TrainDF.drop(["Credit_card"], axis=1)
#print(TrainDF)


# In[5]:


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

print(TestDF_S)


# In[6]:


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
graph.render("Tree_Record5") 


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



# In[11]:


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
                 class_names=['No', 'Yes'])
graph = pydotplus.graph_from_dot_data(dot_data2.getvalue())  
graph.write_png('DecisionTree_FinalExam.png')


# In[12]:


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


# In[13]:


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


# In[14]:


#PRETTY VISUALIZATION WITHOUT SCALING

dot_data2 = StringIO()

export_graphviz(MyDT_R2, out_file=dot_data2,  
                 filled=True, rounded=True,
                 special_characters=True,
                 feature_names = TrainDF.columns,
                 class_names=['No', 'Yes'])
graph = pydotplus.graph_from_dot_data(dot_data2.getvalue())  
graph.write_png('DecisionTree_FinalExam_noscale.png')


# In[17]:


#RandomForest without scaling variables


RF1 = RandomForestClassifier()
RF1.fit(TrainDF, TrainLabels)
RF1_pred=RF1.predict(TestDF)

bn_matrix_RF = confusion_matrix(TestLabels, RF1_pred)
print("\nThe confusion matrix is:")
print(bn_matrix_RF)

#Visualize random forest
Features=TrainDF.columns
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
fig2.savefig('THREEtrees_RF_FinalExam.png')


# In[21]:


#Problem 2: Determine which items to sell to a client
#Method: Association Rule Mining

#!pip install mlxtend

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder  # Import from preprocessing

print(AmazonARM)



# In[32]:


#Restructure data
df = AmazonARM["Most_recent_purchase"].str.split(', ', expand=True).stack()
df = pd.get_dummies(df)
df = df.groupby(level=0).max()
#print(df)

# Create rules
frequent_itemsets = apriori(df, min_support=0.10, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
print(rules.head())

#Plot item frequency
item_frequency = df.sum().sort_values(ascending=False)
item_frequency.head(20).plot(kind="bar", figsize=(10, 6), title="Amazon Purchase Frequency")
plt.ylabel("Frequency")
plt.show()


# In[36]:


#Sort rules

# Sort by support
sorted_rules_support = rules.sort_values(by="support", ascending=False)
#print(sorted_rules_support.head(15))

# Sort by confidence
sorted_rules_confidence = rules.sort_values(by="confidence", ascending=False)
print(sorted_rules_confidence.head(15))

# Sort by lift
sorted_rules_lift = rules.sort_values(by="lift", ascending=False)
#print(sorted_rules_lift.head(15))

# Summary of SortedRulesLift
#print(sorted_rules_lift.describe())

#Compare support, life, and confidence
# SubrulesK
subrulesK = sorted_rules_lift.head(20)
# Plot subrulesK
plt.figure(figsize=(10, 6))
plt.scatter(subrulesK['support'], subrulesK['confidence'], c=subrulesK['lift'], cmap='viridis', s=100)
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("SubrulesK")
plt.colorbar()
plt.show()

