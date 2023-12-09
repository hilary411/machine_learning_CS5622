#ASSOCIATION RULE MINING
#FINAL EXAM
#Hilary Brunberg

#install.packages("arules")
library(arules)
library(arulesViz)
library(tidyr)
library(dplyr)


AmazonDF <- read.csv("~/Desktop/Desktop - MacBook Air (5)/CU Boulder Coursework/Fall 2023/MachingLearning/FinalExam/MLFinalExamData.csv")

#cleaning and preparing data for ARM
Amazon_ARM <- AmazonDF[-c(2, 3, 4)] 

df <- Amazon_ARM %>%
  separate(Most_recent_purchase, into = paste0("Item_", 1:8), sep = ", ", fill = "right")
colnames(df) <- NULL


ARMfile <- "~/Desktop/Desktop - MacBook Air (5)/CU Boulder Coursework/Fall 2023/MachingLearning/FinalExam/ARMData4.csv"

write.csv(df, file = ARMfile, row.names = FALSE)


#creating transactions

transactions <- read.transactions(ARMfile, rm.duplicates=FALSE, format="basket", sep=',', cols=NULL)
inspect(transactions)


#creating rules

rules <- apriori(transactions, parameter = list(supp = 0.10, conf = 0.10, minlen=2))
inspect(rules)

#plotting most frequently transacted items
itemFrequencyPlot(transactions, topN=20, type="absolute")


#sorting rules

#sort by support
SortedRulesSupport <- sort(rules, by="support", decreasing=TRUE)
inspect(SortedRulesSupport[1:15])


#sort by confidence
SortedRulesConf <- sort(rules, by="confidence", decreasing=TRUE)
inspect(SortedRulesConf[1:15])

#sort by lift
SortedRulesLift <- sort(rules, by="lift", decreasing=TRUE)
inspect(SortedRulesLift[1:15])
(summary(SortedRulesLift))


#plotting network of rules

subrulesK <- head(sort(SortedRulesLift, by="confidence"),20)
plot(subrulesK, jitter = 0)

plot(subrulesK, method="graph", engine="interactive")
plot(subrulesK, method="graph", engine="htmlwidget")