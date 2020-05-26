# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 22:18:59 2020

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from urllib.request import urlretrieve
import pandas as pd
n_lists=[]
distances = []
dd=[]
R=[]
t=[]
my_df=[]
neighbors=[]
wk=[]
uu=[]
ll=[]
rr=[]
l=[]

df= pd.read_csv('breast_cancer.csv')
X = df.iloc[:,2:12].values
y = df.iloc[:, 1].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

n_lists = np.append(X, y[:, None], axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test,Y_train, Y_test  = train_test_split(X,y, test_size = 0.25, random_state = 0)
d1=np.append(X_train,Y_train[:, None], axis=1)
def euclidean_distance(row1, row2):
	distance = 0.0
    
	for i in range(len(row1)):
		distance += (row1[i] - row2[i] )**2
	return sqrt(distance)
def get_neighbors(train, test,k):
    
    
    for test_row in test:
        
        dist= euclidean_distance(test_row, train)
#        print(dist)
        distances.append(int( dist))
        
for r in X_test:
    distances.clear()
    ll.clear()
    l.clear()
    
    get_neighbors(r, X_train, 5)
    my_df = pd.DataFrame(d1,distances)
    my_df.to_csv(r'C:\Users\HP\Desktop\SHREYA\New folder\ex.csv',  header=False)
    df1= pd.read_csv('ex.csv')
    l = df1.values.tolist()
    def Sort(l): 
        l.sort(key = lambda x: x[0]) 
        return l 
    Sort(l) 
    ll=l[:(7)]
    freq1=0
    freq2=0
    for jj in range(len(ll)):
        if (ll[jj][11]==0):
            freq1=freq1+1
        else:
            freq2=freq2+1
    if(freq1>freq2):
          rr.append(0)
    else:
            rr.append(1)
    
def getaccuracy(testset,pred):
    correct=0
    for x in range(len(testset)):
        if testset[x]== pred[x]:
            correct+=1
    return (correct/float(len(testset)))*100
acc=getaccuracy(Y_test,rr)
print(acc)     
        
  


