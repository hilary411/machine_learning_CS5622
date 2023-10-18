#ASSOCIATION RULE MINING

#install.packages("arules")
library(arules)
library(arulesViz)

#RULES FOR RESTORATION, GDP PER CAPITA, AND ANIMAL PRODUCTION

Bahia_subset <- Bahia[-c(1, 2, 3, 4, 5, 8, 9, 10, 11)] 
Bahia_ARM <- Bahia_subset

# Convert each column to binary
for (col_name in names(Bahia_ARM)) {
  # Calculate the median of the column
  med <- median(Bahia_ARM[[col_name]], na.rm = TRUE)
  
  # Convert values above the median to "HIGH col_name" and below or equal to median to "LOW col_name"
  Bahia_ARM[[col_name]] <- ifelse(Bahia_ARM[[col_name]] > med, paste("HIGH", col_name), paste("LOW", col_name))
}

transaction_strings <- apply(Bahia_ARM, 1, function(row) {
  paste(as.character(row), collapse = ",")
})

tmp_file <- tempfile()
writeLines(transaction_strings, tmp_file)
transactions <- read.transactions(tmp_file, format = "basket", sep = ",")


rules <- apriori(transactions, parameter = list(supp = 0.20, conf = 0.20, minlen=2))
inspect(rules)
itemFrequencyPlot(transactions, topN=20, type="absolute")

#sort rules
SortedRulesK <- sort(rules, by="lift", decreasing=TRUE)
inspect(SortedRulesK[1:12])
(summary(SortedRulesK))


subrulesK <- head(sort(SortedRulesK, by="lift"),10)
plot(subrulesK, jitter = 0)

plot(subrulesK, method="graph", engine="interactive")
plot(subrulesK, method="graph", engine="htmlwidget")

