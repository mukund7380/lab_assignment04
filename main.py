import pandas as pd;
import numpy as np;
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

all = pd.read_csv('C:\\Users//ASUS//Downloads//AirQuality.csv')
data = all.drop(all.columns[0:all.shape[1]-4],axis=1)
data = data.drop(data.columns[2:4],axis=1)
# print(data)
rh = data.get("RH")
# print(rh)
ah = data.get("AH")
# print(ah)

rh_train,rh_test,ah_train,ah_test = train_test_split(rh,ah,test_size=0.2,random_state=0)

meanRh = np.mean(rh_train)
meanAh = np.mean(ah_train)

num = 0
dem = 0

rh_train_val = rh_train.values
ah_train_val = ah_train.values

for i in range(len(rh_train_val)):
    num+=(rh_train_val[i]-meanRh)*(ah_train_val[i]-meanAh)
    dem+=(rh_train_val[i]-meanRh)**2

slope = num/dem
# print(slope)

intercept = meanAh - (slope*meanRh)
# print(intercept)

ah_pred = slope * rh_test+intercept
# print(ah_pred)

rh_test["predicted"] = ah_pred
rh_test["actual"] = ah_train
# print(rh_test)

score = r2_score(ah_test,ah_pred)
print(score*100)
