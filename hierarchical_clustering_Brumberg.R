# HIERARCHICAL CLUSTERING CODE

library(stats)  
library(NbClust)
library(cluster)
library(mclust)
library(factoextra)

Bahia <- read.csv("~/Desktop/CU Boulder Coursework/Spatial Programming/FinalData/BahiaSubsample.csv")

#GDP PER CAPITA

# Subset data set to create simple dataframe that can be used with clustering
(Bahia_GDP_percap <- Bahia[-c(1, 2, 3, 4, 5, 8, 9, 10, 11)])


## Distance Metric Matrices using dist
(Eucl_Dist_percap <- stats::dist(Bahia_GDP_percap,method="manhattan", p=2))  

library(stylo)
(CosSim <- stylo::dist.cosine(as.matrix(Bahia_GDP_percap)))

Hist1_percap <- stats::hclust(Eucl_Dist_percap, method="ward.D2") #Ward is a general agglomerative 
plot(Hist1_percap, cex=0.9, hang=-1, main = "Cluster Dendorogram (GDP per capita, animal production, and percent area restored)")
rect.hclust(Hist1_percap, k=3)

# Silhouette to determine optimal number of clusters
factoextra::fviz_nbclust(Bahia_GDP_percap, method = "silhouette", 
                         FUN = hcut, k.max = 5)
# 3 is optimal




#OVERALL GDP

# Subset data set to create simple dataframe that can be used with clustering
(Bahia_GDP <- Bahia[-c(1, 2, 3, 4, 6, 8, 9, 10, 11)])


## Distance Metric Matrices using dist
(Eucl_Dist_GDP <- stats::dist(Bahia_GDP,method="manhattan", p=2))  

library(stylo)
(CosSim <- stylo::dist.cosine(as.matrix(Bahia_GDP)))

Hist1_GDP <- stats::hclust(Eucl_Dist_GDP, method="ward.D2") #Ward is a general agglomeration 
plot(Hist1_GDP, cex=0.9, hang=-1, main = "Cluster Dendorogram (GDP, animal production, and percent area restored)")
rect.hclust(Hist1_GDP, k=2)

factoextra::fviz_nbclust(Bahia_GDP, method = "silhouette", 
                         FUN = hcut, k.max = 5)

# 2 is optimal


#JUST AGRICULTURAL PRODUCTION AND RESTORED PERCENT

# Subset data set to create simple dataframe that can be used with clustering
(Bahia_restore <- Bahia[-c(1, 2, 3, 4, 5, 6, 8, 9, 10, 11)])


## Distance Metric Matrices using dist
(Eucl_Dist_restore<- stats::dist(Bahia_restore,method="manhattan", p=2))  

library(stylo)
(CosSim <- stylo::dist.cosine(as.matrix(Bahia_restore)))

Hist1_restore <- stats::hclust(Eucl_Dist_restore, method="ward.D2") #Ward is a general agglomeration 
plot(Hist1_restore, cex=0.9, hang=-1, main = "Cluster Dendorogram (animal production and percent area restored)")
rect.hclust(Hist1_restore, k=2)


factoextra::fviz_nbclust(Bahia_restore, method = "silhouette", 
                         FUN = hcut, k.max = 5)
# 2 is optimal
