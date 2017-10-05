import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats, integrate
import matplotlib.pyplot as plt
import random

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
    '''if data.iat[i,54]==1.0:
        y[i]="Spruce/Fir"
    if data.iat[i,54]==2.0:
        y[i]="Lodgepole Pine"
    if data.iat[i,54]==3.0:
        y[i]="Ponderosa Pine"
    if data.iat[i,54]==4.0:
        y[i]="Cottonwood/Willow"
    if data.iat[i,54]==5.0:
        y[i]="Aspen"
    if data.iat[i,54]==6.0:
        y[i]="Douglas-Fir"
    if data.iat[i,54]==7.0:
        y[i]="Krummholz"'''




#used to test that the operation was sucesful
print("target")
for i in range(0,10):
    print(y[i])

#admend the dataframe to contain the relevant information
data = data.drop(0, axis=1)
data = data.drop(54,axis=1)
#print(data.info())

#look at the distribution of my classes
<<<<<<< HEAD

sns.distplot(y, kde=False, rug=True)
=======
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

#look at the distribution of my classes
sns.distplot(y, kde=False)
>>>>>>> b06512a84878329434382501bd8874770713a553
plt.show()
#Note: There is the argument denoting the location from which the samples were taken,  which we might consider to exclude
#and afterwards apply weights, partucularly to class 4

#look at the distribution of each data set
#based on the distrbution we can find the best metric 

#