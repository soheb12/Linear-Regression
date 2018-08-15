# python3 regression.py

import sys
import pandas as pd
import math , quandl, datetime
import numpy as np
from sklearn import preprocessing , model_selection , svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

#sys.stdout = open("output.txt","w")

style.use('ggplot')

df = quandl.get('WIKI/GOOGL',authtoken = 'JUeJ4SadY3xkwECPHdLi')  #Dataframe is obtained 

df = df[['Adj. Open' , 'Adj. Close' , 'Adj. Low' , 'Adj. High' , 'Adj. Volume']] #just select these collumns from obatained data

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'] ) / df['Adj. Close'] * 100.0   #Adding High-low %age as a feauture
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open'] / df['Adj. Open']) * 100.0 #Adding %age change in closing & opening price as a feauture

df = df[['Adj. Close' , 'HL_PCT' , 'PCT_Change' , 'Adj. Volume']] #just select these collumns from available dataframe
df.fillna(-99999,inplace = True)       #Replace all NA's with -99999

forecast_col = "Adj. Close"                    #This is what we will forecast 
forecast_out =  int(math.ceil( 0.01*len(df) )) #How many days ahead of data you wanna forecast

df['label'] = df[forecast_col].shift(-forecast_out) #give label values of forecast_col but from forecast_out(a number) rows ahead of current row


# Here 'X' is set of Feautures(input) and 'y' is Label(output)
X = np.array(df.drop(['label'] , 1)) #Except Label collumn take all & here 1 indicates collumn , 0 refers to row
y = np.array(df['label'])            #Take Just the label collumn 

X = preprocessing.scale(X) #Scalling down all the feauture values 10,20,30 becomes 1,2,3
X_Lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace = True) #remove all the NA rows
y = np.array(df['label'])

X_train , X_test , y_train , y_test = model_selection.train_test_split(X , y , test_size = 0.2) #getting train,test data (Note : only 20% of of given data is used for testing as test_size = 0.2)


clf = LinearRegression(n_jobs = -1)      #Using LinearRegression as our classifier
clf.fit(X_train , y_train)               #Training our model/classifier with training data
with open('linearregression.pickle','wb') as f:
    pickle.dump(clf,f)


pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test , y_test)    #Testing our model/classifier with testing data
forecast_set = clf.predict(X_Lately)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

