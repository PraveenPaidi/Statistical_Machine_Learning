#!/usr/bin/env python
# coding: utf-8

# In[14]:


from mnist import MNIST
import numpy as np
import math
from matplotlib import pyplot as plt
mndata=MNIST(r'C:\Users\ppaidi')
X_train, labels_train = map(np.array, mndata.load_training())
X_test, labels_test = map(np.array, mndata.load_testing())

#making the values 2 and 7 as -1 and 1
z1, z2 = 2,7 
X_train = np.vstack( (X_train[labels_train==z1], X_train[labels_train==z2]) )
labels_train = np.hstack( (labels_train[labels_train==z1], labels_train[labels_train==z2]) )
X_test = np.vstack( (X_test[labels_test==z1], X_test[labels_test==z2]) )
labels_test = np.hstack( (labels_test[labels_test==z1], labels_test[labels_test==z2]) )

X_train = X_train/255.0
X_test = X_test/255.0
(m,n)=X_train.shape
(m1,n1)=X_test.shape

#assigning the y_values 
l_train=np.zeros(m)
l_test=np.zeros(m1)
l_train[labels_train==z1] = -1
l_train[labels_train==z2] = 1
l_test[labels_test==z1] = -1
l_test[labels_test==z2] = 1
Y=l_train
Y1=l_test
yhat_train=np.zeros(m)
yhat_test=np.zeros(m1)

#values initiation
lam=0.1
N= 0.01
w=np.zeros(n)
b=np.zeros(m)
b1=np.zeros(m1)
wt=np.zeros(n)
bt=np.zeros(m)
bt1=np.zeros(m1)
sum=0
p1=[]
cost=[]
cost1=[]
count1=[]
count=0
I=np.identity(784)
lamb=0.0001
converged= False
J=0
J1=0
mis_error=[]
mis_error1=[]

#loop for convergence
while not converged:
    
    U=1/(1+np.exp(np.multiply(-Y,(b+np.dot(X_train,w)))))
    wt=w-N*((np.sum(X_train*(np.multiply((U-1),Y))[:, np.newaxis],axis=0))/m +2*lam*w)
    epsilon=np.absolute(wt-w)
    w=wt[:]
    
    #train b 
    U=1/(1+np.exp(-np.multiply(Y,(b+np.dot(X_train,w)))))
    bt=b - N*(U-1)*Y/m
    epsilon1=np.absolute(bt-b)
    b=bt[:]
    
    #test b
    U1=1/(1+np.exp(-np.multiply(Y1,(b1+np.dot(X_test,w)))))
    bt1=b1 - N*(U1-1)*Y1/m1
    b1=bt1[:]
    
    converged=True
    
    for i in range(n):
        if epsilon[i]>0.0001:
            converged=False
            break
    
    #cost function of training data
    J=lam*np.linalg.norm(w, ord=2)**2 +(np.sum(np.log(1+np.exp(np.multiply(-Y,(b+np.dot(X_train,w)))))))/m 
    
    #cost function of test data
    J1=lam*np.linalg.norm(w, ord=2)**2 +(np.sum(np.log(1+np.exp(np.multiply(-Y1,(b1+np.dot(X_test,w)))))))/m1 
    
    # y_hat train calc
    yhat_train=b+np.dot(X_train,w)
    mis=0
    
    # y_hat test calc
    yhat_test=b1+np.dot(X_test,w)
    mis1=0 

    #calc of error for train
    for i in range(m):
        if yhat_train[i]*Y[i]<0:
            mis=mis+1
    
    #calc of error for test
    for i in range(m1):
        if yhat_test[i]*Y1[i]<0:
            mis1=mis1+1      
    
    mis_error.append(mis/m)    # appending the train_error
    mis_error1.append(mis1/m1) # appending the test_error
    
 
    count=count+1
    
    cost.append(J)
    cost1.append(J1)
    count1.append(count)
    
fig1=plt.figure("Plot1")
plt.plot(count1,cost, label='Training set Cost function')
plt.plot(count1,cost1, label='Test set Cost function')
plt.xlabel('Number of Iterations')
plt.ylabel('Cost function')
plt.title("Plot of Iterations vs Cost Function")
plt.legend()
plt.show()

fig2=plt.figure("plot")
plt.plot(count1,mis_error,label='train_error')
plt.plot(count1,mis_error1,label='test_error')
plt.xlabel('Number of Iterations')
plt.ylabel('Mis_error')
plt.legend()
plt.show() 


# In[ ]:




