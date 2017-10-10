import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats, integrate
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

#gets our data set in a usable form
data = pd.read_csv('usable_covtype.txt', header= None, )

#check to see what we get
#print(data.info())
#print(df.columns)

#note an index column was added that will have to be removed

#creates an array of our target values
y=np.empty(581012, dtype=int)
for i in range(0,581012):
    y[i] = data.values[i,54]






#admend the dataframe to contain the relevant information
data = data.drop(0, axis=1)
data = data.drop(54,axis=1)
#print(data.info())

#look at the distribution of my classes

#sns.distplot(y, kde=False)
#plt.show()

new_data_list=[]
new_y_list=[]

for j in range(0,len(y)):
    randNum=0
    limit =1.0
    if y[j]==1:
        randNum = random.random()
        limit=50000/211840
    if y[j]==2:
        randNum = random.random()
        limit=50000/283301

    if randNum<limit:
        new_data_list.append(data.values[j])
        new_y_list.append(y[j])

data=np.asarray(new_data_list)
y=np.asarray(new_y_list)


#split the data into traing and test
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.20)

scaler_x = preprocessing.StandardScaler().fit(X_train)
X_train = scaler_x.transform(X_train)
X_test = scaler_x.transform(X_test)
'''
next we will try knearest neighber because it should be fairly simple and quick to run and because it is the only 
one that does not generate a model may give us a unique perspective
'''

klist = []
acclist = []
highest_accuracy=0
k_highest_accuracy = 0
for k in range(1,15):
    print(k)
    kNeighClassif = KNeighborsClassifier(n_neighbors=(k), weights= 'uniform', algorithm='kd_tree', n_jobs=-1)
    kNeighClassif.fit(X_train,y_train)
    print(kNeighClassif.score(X=X_train,y=y_train))
    print(kNeighClassif.score(X=X_test,y=y_test))
    currAccuracy = kNeighClassif.score(X=X_test,y=y_test)
    klist.append(k)
    acclist.append(currAccuracy)
    #print("Accuracy for k=" + str(k) +" :" + str(currAccuracy))
    if(currAccuracy>highest_accuracy):
        highest_accuracy = currAccuracy
        k_highest_accuracy =k
print("kNN-Classifier: Highest test accuracy was for k=" + str(k_highest_accuracy) + " :" + str(highest_accuracy))

'''
due to the cost associated with running this, which makes sense, it would be too long to run through 15 trials, however
after the first 6 it was possible to note a pattern of decreasing k. This means k=3 will be our best value
'''