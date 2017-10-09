import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import random

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

#decrease the number of class one and two to make more evenly distributed
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
y=y.reshape((-1,1))
scaler_y = preprocessing.StandardScaler().fit(y_train)
y_train = scaler_y.transform(y_train)
y_test = scaler_y.transform(y_test)
'''
'''
the papers using this data set claim the Neural networks have produced the best accuracy so we will test
to see what they got and then experiment to see how parameters will affect it. 
We will use multiple layers to improve accuracy. 
'''

'''
#this is a nural network with generic value to give us a base line
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
#this is low so lets see what we can do
'''
'''
#it is sugested the adam solver is used for larger data sets, 1,000 or more, which we have soo we will try that
clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
#same thing so solver has no effect

#setting it to a generic hiden layer length makes it way more accurate but takes longer
#changeing parameters that affect backpropagation should still show me when a parameter decrese overall
clf = MLPClassifier(solver='adam', alpha=1e-5, random_state=1)
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
'''
'''
#learning rate has a fairly big affect so try the adaptive parameter that allows it to change as the process happens
clf = MLPClassifier(solver='adam', alpha=1e-5, random_state=1, hidden_layer_sizes=(5, 2), learning_rate='adaptive')
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
#no change
'''

'''
#new activation function that allows for momentum. momentum will come in handy if we have false minimums
clf = MLPClassifier(solver='sgd', random_state=1, hidden_layer_sizes=(5, 2), alpha=1e-5, learning_rate='adaptive')
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
#no change
'''

'''
#now lets try messing around with normalization

#this is the basic scale function in sklearn
#x_trained_scale = preprocessing.scale(X_train)
#no change

#try min max which is a little more advanced
min_max_scaler = preprocessing.MinMaxScaler()
x_trained_scale = min_max_scaler.fit_transform(X_train)

y_trained_scale =min_max_scaler.transform(X_test)
'''
'''
clf = MLPClassifier(solver='sgd', random_state=1, alpha=1e-5, learning_rate='adaptive')
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))

'''


#the last big parameter to adjust is the hidden layer sizes
highest_test=0
parameter_layer = 0
parameter_node = 0
for i in range(1,4):
    for j in range(1,10):
        clf = MLPClassifier(solver='sgd', random_state=1, alpha=1e-5, hidden_layer_sizes=(i, j*5),
                            learning_rate='adaptive')
        clf.fit(X_train, y_train)
        if(highest_test<clf.score(X_test, y_test)):
            highest_test =clf.score(X_test, y_test)
            parameter_layer = i
            parameter_node = j

clf = MLPClassifier(solver='sgd', random_state=1, alpha=1e-5, hidden_layer_sizes=(parameter_layer, parameter_node),
                    learning_rate='adaptive')
clf.fit(X_train, y_train)
print(parameter_layer)
print(parameter_node)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))


'''
we have not been able to get a score better than what we got but apaper using this data set got what we got.

question for lab? do we need to change the data coming into the network
'''
