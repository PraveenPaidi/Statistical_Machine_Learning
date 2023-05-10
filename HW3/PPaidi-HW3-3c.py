#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
n=100
z=np.zeros((2,n))
x=np.zeros((2,n))
b=np.array([[1],[2]])
A=np.array([[1,0],[0,1.414]])
for i in range(100):
    z[:,i]=np.random.randn(2)
x=np.dot(A,z)+b
U=np.zeros((2,1))

U=np.mean(x,axis=1)
U=U.reshape(2,1)
E=np.zeros((2,2))
E=np.dot((x-U),(x-U).T)/(n-1)
evalue, evect = np.linalg.eig(E)
V = np.array(evect)

X_tilda=np.zeros((2,n))

evalue=evalue.reshape(2,1)
evalue=np.sqrt(evalue)
for i in range(2):
    X_tilda[i,:]=(np.dot(V[:,i].T,(x-U))/evalue[i]).T
fig1=plt.figure("   ")

plt.scatter(X_tilda[1,:],X_tilda[0,:],label='scatter plot of 1st distribution')
plt.legend() 
plt.xlabel('X1')
plt.ylabel('X2')   
plt.title('X_tilda distribution 1')
plt.show()

#second set


x1=np.zeros((2,n))
b1=np.array([[2],[-2]])
A1=np.array([[3,1],[1,2]])
A1=sqrtm(A1)
x1=np.dot(A1,z)+b1

U1=np.mean(x1,axis=1)
U1=U1.reshape(2,1)
E1=np.zeros((2,2))

E1=np.dot((x1-U1),(x1-U1).T)/(n-1)
evalue1, evect1 = np.linalg.eig(E1) 
V1 = np.array(evect1)
X_tilda1=np.zeros((2,n))
evalue1=evalue1.reshape(2,1)
evalue1=np.sqrt(evalue1)



for i in range(2):
    X_tilda1[i,:]=(np.dot(V1[:,i].T,(x1-U1))/evalue1[i]).T
fig2=plt.figure("   ")
plt.scatter(X_tilda1[0,:],X_tilda1[1,:],label='scatter plot of 2nd distribution')
plt.legend() 
plt.xlabel('X1')
plt.ylabel('X2')   
plt.title('X_tilda distribution 2')
plt.show()


# In[ ]:





# In[ ]:




