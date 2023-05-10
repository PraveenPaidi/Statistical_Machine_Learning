#!/usr/bin/env python
# coding: utf-8

# In[5]:


from mnist import MNIST
import numpy as np
from matplotlib import pyplot as plt
I=np.identity(784)
lamb=0.0001
k=10
mndata=MNIST(r'C:\Users\praveen')
X_train, labels_train = map(np.array, mndata.load_training())
X_test, labels_test = map(np.array, mndata.load_testing())
X_train = X_train/255.0
X_test = X_test/255.0

#hot encoding Y 
k1= np.array(labels_train).reshape(-1)
Y= np.eye(k)[k1]

# train function
def train(X_train,Y):
    a=np.linalg.solve(((np.dot(X_train.T,X_train)+np.dot(lamb,I))),I)
    w_cap=np.dot(a,np.dot(X_train.T,Y))
    return w_cap

#predict function
def predict(w_cap,X_train):
    p=[]
    predict=np.dot(X_train,w_cap)
    (a,b)=np.shape(X_train)
    for i in range(a):
        p.append(np.argmax(predict[i]))
    return p

#error function
def error(p,X_train,labels_train):
    sum=0
    (a,b)=np.shape(X_train)
    for i in range(a):
        if p[i]!=labels_train[i]:
            sum=sum+1
    return (sum,a)

#error_percentage function
def error_percentage(train_error,a):      
    e=((100*train_error/a))
    return e
    

#error percentage of training data
w_cap=train(X_train,Y)     # use same w_cap for test and training 
p=predict(w_cap,X_train)
(train_error,a)=error(p,X_train,labels_train)
e1=error_percentage(train_error,a)
print('train error is ',e1,'percentage')



#error percentage of test data
p=predict(w_cap,X_test)
(test_error,a)=error(p,X_test,labels_test)
e2=error_percentage(test_error,a)
print('test error is ',e2,'percentage')









# In[ ]:




