#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
n=100
z=np.zeros((2,n))
x=np.zeros((2,n))
b=np.array([[1],[2]])
A=np.array([[1,0],[0,2]])
A=sqrtm(A)

for i in range(100):
    z[:,i]=np.random.randn(2)

fig1=plt.figure("l")
x=np.dot(A,z)+b[:]
for i in range(100):
    plt.scatter(x[0,i],x[1,i],marker='^')
plt.xlim(-7,7)
plt.ylim(-7,7)    
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Scatter plot of mean 1 and covariance 1 ')


x1=np.zeros((2,n))
b1=np.array([[2],[-2]])
A1=np.array([[3,1],[1,2]])
A1=sqrtm(A1)


fig2=plt.figure("lol")
x1=np.dot(A1,z)+b1[:]
for i in range(100):
    plt.scatter(x1[0,i],x1[1,i],marker='^')
plt.xlim(-7,7)
plt.ylim(-7,7)
plt.xlabel('X1')
plt.ylabel('X2')   
plt.title('Scatter plot of mean 2 and covariance 2 ')

