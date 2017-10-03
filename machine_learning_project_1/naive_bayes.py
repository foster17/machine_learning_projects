import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing


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

#split the data into traing and test
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.20, random_state=42)

'''
we will try naive bayes because it should be fairly simple to learn and will give even more explanation to 
our class distribution 
'''
#because we are hoping to understand the probability with different distributions of classes it may need be necessary
#to normalise the data. The min max should work well for this(this is assuming no outliers so I will need to check)

#min_max_scaler = preprocessing.MinMaxScaler()
#will aply this when I figure out what is wrog in NN

naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train)
print("Naive Bayes Classifier: Train Accuracy: " + str(naive_bayes.score(X_train, y_train)))
print("Naive Bayes Classifier: Test Accuracy: " + str(naive_bayes.score(X_test, y_test)))
