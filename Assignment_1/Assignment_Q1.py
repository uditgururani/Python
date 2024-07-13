# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 06:06:56 2023

@author: DELL
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

dict1 = {"Panel_number":[1,2,3,4,5,6,7,8,9,10],"Film_thickness":[2.7,4.2,5.1,3.1,2.5,2.9,3.7,4.1,2.7,3.6],
         "Average_roughness":[0.26,0.31,0.29,0.35,0.21,0.33,0.39,0.29,0.18,0.21],
         "Current":[1.13,1.51,1.67,1.23,1.38,1.13,1.68,1.57,1.44,1.58]}
df1 = pd.DataFrame(dict1)
#print(df)
#df.describe()
print(df1.describe())
def mean(x):
    S = 0
    for i in range(len(x)):
        S+=x[i]
    
    a = S/len(x)
    return a

def median(x):
    x.sort()
    N = len(x)
    if (N%2==0):
        median = (x[N//2] + x[N//2 +1])/2
    else :
        median = x[N//2]
        
    return median

def stdDev(x):
    avg = mean(x)
    S = 0
    for i in range(len(x)):
        S+= (x[i]-avg)**2
    variance = S/(len(x)-1)
    stdDev = np.sqrt(variance)
    
    return stdDev

def kurtosis(x):
    n = len(x)
    avg = mean(x)
    std = stdDev(x)
    s = 0
    for i in range(n):
        s = s+(avg-x[i])**4/std**4
    K = n*(n+1)/((n-1)*(n-2)*(n-3))*s - 3*(n-1)**2/((n-2)*(n-3))
    return K

def zscore(x):
    z = np.zeros(len(x))
    avg = mean(x)
    std = stdDev(x)
    for i in range(len(x)):
        z[i] = ((x[i]-avg)/std)
    
    return z

dict2 = {"Panel_number":[1,2,3,4,5,6,7,8,9,10],
        "Film_thickness": zscore(df1["Film_thickness"]),
        "Average_roughness": zscore(df1["Average_roughness"]),
        "Current": zscore(df1["Current"])}
df2 = pd.DataFrame(dict2)
print(df2.describe())
print(df2)

print(kurtosis(df2["Film_thickness"]))
print(kurtosis(df2["Average_roughness"]))
print(kurtosis(df2["Current"]))

plt.axhline(y=0,color = 'Black')
plt.axvline(x = 0,color = 'Black')
plt.title('Histogram')
plt.xlabel('Film thickness')
plt.ylabel('Frequency')
plt.hist(df2["Film_thickness"])
plt.show()

plt.axhline(y=0,color = 'Black')
plt.axvline(x = 0,color = 'Black')
plt.title('Histogram')
plt.xlabel('Average roughness')
plt.ylabel('Frequency')
plt.hist(df2["Average_roughness"])
plt.show()

plt.axhline(y=0,color = 'Black')
plt.axvline(x = 0,color = 'Black')
plt.title('Histogram')
plt.xlabel('Current')
plt.ylabel('Frequency')
plt.hist(df2["Current"])
plt.show()

plt.axhline(y=0,color = 'Black')
plt.axvline(x = 0,color = 'Black')
plt.grid()
plt.title('Scatter Plot')
plt.xlabel('Film thckness')
plt.ylabel('Average roughness')
plt.scatter(df1["Film_thickness"],df1["Average_roughness"])
plt.show()

plt.axhline(y=0,color = 'Black')
plt.axvline(x = 0,color = 'Black')
plt.grid()
plt.title('Scatter Plot')
plt.xlabel('Average roughness')
plt.ylabel('Current')
plt.scatter(df1["Average_roughness"],df1["Current"])
plt.show()

plt.axhline(y=0,color = 'Black')
plt.axvline(x = 0,color = 'Black')
plt.grid()
plt.title('Scatter Plot')
plt.xlabel('Film thckness')
plt.ylabel('Current')
plt.scatter(df1["Film_thickness"],df1["Current"])
plt.show()

plt.axhline(y=0,color = 'Black')
plt.axvline(x = 0,color = 'Black')
plt.grid()
plt.title('Scatter Plot')
plt.xlabel('Film thckness')
plt.ylabel('Average roughness')
plt.scatter(df2["Film_thickness"],df2["Average_roughness"])
plt.show()

plt.axhline(y=0,color = 'Black')
plt.axvline(x = 0,color = 'Black')
plt.grid()
plt.title('Scatter Plot')
plt.xlabel('Average roughness')
plt.ylabel('Current')
plt.scatter(df2["Average_roughness"],df2["Current"])
plt.show()

plt.axhline(y=0,color = 'Black')
plt.axvline(x = 0,color = 'Black')
plt.grid()
plt.title('Scatter Plot')
plt.xlabel('Film thckness')
plt.ylabel('Current')
plt.scatter(df2["Film_thickness"],df2["Current"])
plt.show()

covmat = np.zeros((3,3))
corrmat = np.zeros((3,3))

def subcov(x1,x2):
    n = len(x1)
    avg1 = mean(x1)
    avg2 = mean(x2)

    t = 0
    for i in range(n):
        t = t+(x1[i]-avg1)*(x2[i]-avg2)
    r = t/(n-1)
    
    return r

    
def cov(df):
    for i in range(3):
        for j in range(3):
            
            covmat[i][j] = subcov(df.iloc[0:,i+1],df.iloc[0:,j+1])
    
    return cov

def subcorr(x1,x2):
    n = len(x1)
    avg1 = mean(x1)
    avg2 = mean(x2)
    std1 = stdDev(x1)
    std2 = stdDev(x2)
    t = 0
    for i in range(n):
        t = t+(x1[i]-avg1)*(x2[i]-avg2)
    r = t/((n-1)*std1*std2)
    
    return r

def corr(df):
    for i in range(3):
        for j in range(3):
            
            corrmat[i][j] = subcorr(df.iloc[0:,i+1],df.iloc[0:,j+1])
    
    return cov
df3 = df1.iloc[0:,1:4]

#x = df3.cov()
#print(df3.cov())

print(cov(df1))
print(corr(df1))
    
