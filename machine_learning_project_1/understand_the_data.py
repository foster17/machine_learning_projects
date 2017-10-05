import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats, integrate
import matplotlib.pyplot as plt


#gets our data set in a usable form
data = pd.read_csv('usable_covtype.txt', header= None, )
#check to see what we get
#print(data.info())
#print(df.columns)

#note an index column was added that will have to be removed

#creates an array of our target values
y=np.empty([581012])
for i in range(0,581012):
    y[i] = data.iat[i,54]
#used to test that the operation was sucesful
#print("target")
#for i in range(0,10):
    #print(y[i])

#ademnd the dataframe to contain the relevant information
data = data.drop(0, axis=1)
data = data.drop(54,axis=1)
#print(data.info())

#look at the distribution of my classes

sns.distplot(y, kde=False, rug=True)
plt.show()
#why wont it load? thi will help to find out what kind of metrics to use

#look at the distribution of each data set
#based on the distrbution we can find the best metric 

