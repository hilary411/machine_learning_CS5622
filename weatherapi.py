#Machine Learning API

import requests  #to query the API 
import re  #regular expressions
import pandas as pd   # for dataframes

from sklearn.feature_extraction.text import CountVectorizer   
#for text vectorization

End="http://api.weatherapi.com/v1/forecast.json?"


#Create URL in dictionary 

URLPost = {'key':'4f1d958451f84d57a6235304231909',
                    'days': 10,
                    'q': 'Brasilia', 
                    'aqi': 'no'}

print(URLPost) #prints as a dictionary

response1=requests.get(End, URLPost)
print(response1)

jsontxt = response1.json()
#print(jsontxt)

## Create a new csv file to save the headlines
filename="brasiliaweather.csv"


MyFILE=open(filename,"w")
### Place the column names in - write to the first row
WriteThis="Date,maxtemp_f,avgtemp_f,totalprecip_in,daily_chance_of_rain\n"
MyFILE.write(WriteThis)
MyFILE.close()


## Open the file for append
MyFILE=open(filename, "a")

## Go through the json text:
for items in jsontxt["forecast"]["forecastday"]:
   # print(items, "\n\n\n")


    Date=items["date"]
    #print(Date)
    
    maxtemp_f=items["day"]["maxtemp_f"]
    #print(maxtemp_f)
    
    avgtemp_f=items["day"]["avgtemp_f"]
    #print(avgtemp_f)
    
    totalprecip_in=items["day"]["totalprecip_in"]
    #print(totalprecip_in)
    
    daily_chance_of_rain=items["day"]["daily_chance_of_rain"]
    #print(daily_chance_of_rain)
    

    WriteThis=str(Date)+","+str(maxtemp_f)+","+ str(avgtemp_f) + "," + str(totalprecip_in) + ","+ str(daily_chance_of_rain) +"\n"
    
    MyFILE.write(WriteThis)

    MyFILE.close()
