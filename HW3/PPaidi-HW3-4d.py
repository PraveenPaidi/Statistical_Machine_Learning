#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

#taking 1000 samples 
n=1000
z=np.zeros((2,n))
x2=np.zeros((2,n))
b=np.array([[1],[1]])
A=np.array([[1,0],[0,2]])
A=sqrtm(A)
for i in range(n):
    z[:,i]=np.random.randn(2)
x2=np.dot(A,z)+b     # calculating for y=-1

# taking 1000 samples
z1=np.zeros((2,n))
x1=np.zeros((2,n))
b1=np.array([[-1],[1]])
A1=np.array([[1,0],[0,1.414]])
for i in range(n):
    z1[:,i]=np.random.randn(2)
x1=np.dot(A1,z1)+b1    # calculating for y=1

#taking y values
y1=np.full(
  shape=1000,
  fill_value=-1,
  dtype=int
)
y1=y1.reshape(n,1)

#taking y values
y2=np.full(
  shape=1000,
  fill_value=1,
    dtype=int
)
y2=y2.reshape(n,1)

#combining the samples data
x=np.concatenate((x1, x2), axis=1)
y=np.concatenate((y2, y1), axis=0)
lam=0.01
converged=False
b=0
N=0.1
count=0
bias=0
sum=np.array([[0],[0]])

# gradient descent for 100 times
for i in range(100):
    w=np.random.rand(2,1) 
    b=np.random.rand(1)
    converged=False  
    #convergence of gradient descent
    while not converged:
        U=1/(1+np.exp(-y*(b+np.dot(x.T,w))))
        lo=(x.T*((U-1)*y))
        p=np.sum(lo,axis=0).reshape(2,1)/2000 +2*lam*w
        wt=w-N*(p)
        epsilon=np.absolute(wt-w)
        epsilon=epsilon.reshape(-1) 
        w=wt[:]            # updating the weights
        
        U=1/(1+np.exp(-y*(b+np.dot(x.T,w))))
        bt=b - N*(U-1)*y/10000
        b=bt[:]            # updating the bias
        converged=True
        for i in range(2):
            if epsilon[i]>0.00001:    # delta
                converged=False
                break
                
        #cost function   # cehecking cost whether it is decreasing
        J=lam*np.linalg.norm(w, ord=2)**2 +(np.sum(np.log(1+np.exp(np.multiply(-y,(b+np.dot(x.T,w)))))))/2000
        bia=np.mean(b,axis=0)
        
    sum=sum+w        # sum is optimized weights
    count=count+1    # counting number of iterations
    bias=bia+bias   
bias=bias/100  
sum=sum/100

w_erm=sum
yhat=[]
opt=1/(1+np.exp(-((np.dot(x.T,sum))+bias)))
for i in range(2000):
    if opt[i]<0.5:
        yhat.append(-1)
    if opt[i]>0.5:
        yhat.append(1)       
accuracy=0
for i in range(2000):
    if y[i]==yhat[i]:
        accuracy=accuracy+1
accuracy=accuracy*100/2000
print('Accuracy of ERM logistic classifier is',accuracy)
bayes_opt=np.array([[-4],[0]])
bias=0
opt=1/(1+np.exp(-(np.dot(x.T,bayes_opt))+bias))
yhat1=[]
for i in range(2000):
    if opt[i]<0.5:
        yhat1.append(-1)
    if opt[i]>0.5:
        yhat1.append(1)

accuracy=0
for i in range(2000):
    if y[i]==yhat1[i]:
        accuracy=accuracy+1
accuracy=accuracy*100/2000
print('Accuracy of Bayesian optimal Model is',accuracy)  


# In[ ]:




