#!/usr/bin/env python
# coding: utf-8

# In[5]:


from mnist import MNIST
import numpy as np
import math
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
mndata=MNIST(r'C:\Users\ppaidi')
X_train, labels_train = map(np.array, mndata.load_training())
X_test, labels_test = map(np.array, mndata.load_testing())
X_train = X_train/255.0
X_test = X_test/255.0
(n,d)=np.shape(X_train)
(n1,d)=np.shape(X_test)

W = np.zeros((d,10))
W_test= np.zeros((d,10))

# one hot encoding of y train 
nb_classes = 10
targets = np.array(labels_train).reshape(-1)
y= np.eye(nb_classes)[targets]

# one hot encoding of y test
nb_classes = 10
targets = np.array(labels_test).reshape(-1)
y_test= np.eye(nb_classes)[targets]

#loop for convergence
converged=False
N=0.01
epsilon1=np.zeros((d,10))
error1=[]
error1_test=[]
count1=[]
count=0


while not converged:
    
    #train data
    wt2=W+(N*(np.dot(X_train.T,(y-(np.dot(X_train,W)))))/n)  # one hot encoded 
    epsilon1=np.absolute(wt2-W) 
    W = np.copy(wt2)

    converged=True  
    converged=(np.abs(epsilon1) < 0.0001).all()
       
    #classification error train
    Y_hat1=np.dot(X_train,W)
    Y_hat11=np.argmax(Y_hat1,axis=1)
    error=0
    
    for i in range(n):
        if Y_hat11[i]!=labels_train[i]:
            error=error+1
            
    count=count+1
    error=(error*100/n)
    
    #classification error test
    Y_hat1_test=np.dot(X_test,W)
    Y_hat11_test=np.argmax(Y_hat1_test,axis=1)
    error1=0
    
    for i in range(n1):
        if Y_hat11_test[i]!=labels_test[i]:
            error1=error1+1
    error1=error1*100/n1
accuracy=100-error
accuracy1=100-error1
print('Training Accuracy for J function is ',accuracy)
print('Test Accuracy for J function is ',accuracy1) 

count=0
# W1 = np.zeros((d,10),dtype=int)
w=np.zeros(d, dtype=int)
nb_classes = 10
targets = np.array(w).reshape(-1)
W1= np.eye(nb_classes)[targets]
N=0.00001

def softmax(x):
    e_x=np.exp(x)
    denom=np.reshape((np.sum(e_x,axis=1)),(n,1))
    return e_x/denom

converged=False
errorcomp=0
while not converged:
    q21=softmax(np.dot(X_train,W1))
    wt4=W1+(N*(np.dot(X_train.T,(y-(q21)))))  # one hot encoded 
    epsilon2=np.absolute(wt4-W1) 
    W1 = np.copy(wt4)
        
    converged=True
    converged=(np.abs(epsilon2) < 0.0001).all()
    check=np.mean(np.abs(epsilon2))
    if check<0.00001 or count==2500:
        converged=True
    
    #classification error
    Y_hatL1=np.dot(X_train,W1)
    Y_hatL11=np.argmax(Y_hatL1,axis=1)
    error=0
    for i in range(n):
        if Y_hatL11[i]!=labels_train[i]:
            error=error+1       
    count=count+1
    error=error*100/n
    
    #classification error test
    Y_hatL1_test=np.dot(X_test,W1)
    Y_hatL11_test=np.argmax(Y_hatL1_test,axis=1)
    error1=0
    for i in range(n1):
        if Y_hatL11_test[i]!=labels_test[i]:
            error1=error1+1 
    error1=error1*100/n1
    
accuracy=100-error
accuracy1=100-error1    
print('Training Accuracy for L function is ',accuracy)
print('Test Accuracy for L function is ',accuracy1) 
    


# In[ ]:




