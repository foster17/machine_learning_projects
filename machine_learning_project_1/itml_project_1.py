import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn import tree
import random
from sklearn.ensemble import RandomForestClassifier


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

#adjust for over representaion of the first two classe

#ademnd the dataframe to contain the relevant information
data = data.drop(0, axis=1)
data = data.drop(54,axis=1)
#print(data.info())

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
'''
kf = KFold(n_splits=10)
train_accuracy = []
test_accuracy = []
'''

'''
by creating  decision tre and veiwing the results it will be possible to get a rough estimate of the affects
each atribute has on makeing the decision and will also alow us to see if this is the best classifier. 
The max depth appears to have the greater affect on the tree, because it can affect all of the other parameters. 
we will try cross validation to give us increasing potential max depths and take the best. 
graphviz could graph this to potentialy provide more information but I can't figure out how to get the modulo.
'''
#see how good the test is before I begin messing with parmaeters and our weighted data
dt = tree.DecisionTreeClassifier(class_weight ={4: 2, 5:3, 6:1.5, 7:1.25})
dt.fit(X_train, y_train)
print(dt.score(X_train, y_train))
print(dt.score(X_test, y_test))
#could be more accurate if I can figure out how to get the crossvalidation working
'''
for train, test in kf.split(X_train):
    print("%s , %s" % (train, test))
    X_cross_train = [X_train[i] for i in train]
    y_cross_train = [y_train[i] for i in train]
    X_cross_test = [X_train[i] for i in test]
    y_cross_test = [y_train[i] for i in test]
'''

highest_test = 0
parameter_depth = 0

''''
for i in range(1,10):
    dt = tree.DecisionTreeClassifier(max_depth=(i*5), class_weight ={4: 2, 5:3, 6:1.5, 7:1.25})
    dt.fit(X_train, y_train)

    train_acc = dt.score(X_train, y_train)
    test_acc = dt.score(X_test, y_test)

    if(test_acc>highest_test):
        highest_test = test_acc
        parameter_depth = i*5
highest_test = 0
'''
<<<<<<< HEAD
'''
for j in range(-10,10,0.1):
    clf = tree.DecisionTreeClassifier(min_impurity_decrease=(j) , class_weight={4: 10, 5:3, 6:1.5, 7:1.25 })
=======
test_acc_list = []
train_acc_list = []
jten_list = []
for j in range(0,100):
    jten = j/1000
    clf = tree.DecisionTreeClassifier(min_impurity_decrease=jten , class_weight={4: 10, 5:3, 6:1.5, 7:1.25 })
>>>>>>> a96b397db46169caa66cece2c6d329dea659879c
    clf.fit(X_train, y_train)
    train_acc =clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    test_acc_list.append(test_acc)
    train_acc_list.append(train_acc)
    jten_list.append(jten)
    if (test_acc > highest_test):
        highest_test = test_acc
        parameter_impurity = jten
fclf = tree.DecisionTreeClassifier(min_impurity_decrease=parameter_impurity,
                                  class_weight={4: 10, 5:3, 6:1.5, 7:1.25 })
fclf.fit(X_train, y_train)
print(fclf.score(X_train, y_train))
print(fclf.score(X_test, y_test))
print(parameter_impurity)
<<<<<<< HEAD
=======
#tried range (1,10,1), best result 0.26...
#tried range (0,20,0.1) best result 0.1/0.68 at 0, all else 0.26
#tried range (0,0.1,0.001)

plot = sns.stripplot(x=2*jten_list, y=train_acc_list+test_acc_list, hue=100*["Training"]+100*["Test"])
plt.xlabel("Minimum Impurity Decrease")
plt.ylabel("Accuracy")
plt.show()
>>>>>>> a96b397db46169caa66cece2c6d329dea659879c
'''
'''
all of the hyper parameters ultimetly have something to do with the length of the tree. min_impurity_decrease
appears to have the most efficient effect. This is because it decides when to stop bassed on a specific value 
'''
'''
this results in an interesting outcome. when we try verying parameter values we instantly drop bellow the 
orriginal test score. This leads implies to us that it would be better to do the default value but overfitting
is still present so we need to try other models
'''
'''
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("data")
'''
'''
it may be possible to improve our tree using bagging and or boosting. boosting will probably be more expensive 
so we will do bagging just to get an understanding of these techinique but given more time we would try boosting 
as well. To do this we will use the random forst program in sklearn
'''
'''
note see what we did before
'''
highest_test = 0
parameter_depth = 0
for j in range(-10,10,0.1):
    clf = RandomForestClassifier(n_estimators=10, min_impurity_decrease=(int(j)) , class_weight={4: 10, 5:3, 6:1.5, 7:1.25 })
    clf.fit(X_train, y_train)
    train_acc =clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(clf.score(X_train, y_train))
    print(clf.score(X_test, y_test))
    if (test_acc > highest_test):
        highest_test = test_acc
        parameter_impurity = j
fclf = tree.DecisionTreeClassifier(min_impurity_decrease=parameter_impurity,
                                  class_weight={4: 10, 5:3, 6:1.5, 7:1.25 })
fclf.fit(X_train, y_train)
print(fclf.score(X_train, y_train))
print(fclf.score(X_test, y_test))
print(parameter_impurity)

'''
for simplicity sake we will just use 10 estimators but in the future this could be a prameter to mess with too
'''