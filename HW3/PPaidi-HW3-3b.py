#!/usr/bin/env python
# coding: utf-8

# In[32]:


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
    
o1=(U[0,0])
o2=(U[1,0])
fig1=plt.figure("   ")
ax = plt.axes()
ax.arrow(o1,o2, V[0,0],V[1,0] , head_width=0.5, head_length=0.5,color='black')
plt.ylim(-5,5)
plt.xlim(-5,5)

ax.arrow(o1,o2, V[0,1],V[1,1] , head_width=0.5, head_length=0.5, color='black',label='eigen Vectors')

plt.scatter(x[0,:],x[1,:],marker='^',label='scatter plot of 1st disctribution')
plt.legend() 
plt.xlabel('X1')
plt.ylabel('X2')   
plt.title('Eigen vectors along with distribution 1')
plt.show()



#for the second plot

for i in range(100):
    z[:,i]=np.random.randn(2)
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
   
o11=(U1[0,0])
o22=(U1[1,0])


fig1=plt.figure("   ")
ax = plt.axes()
ax.arrow(o11,o22, V1[0,0],V1[1,0] , head_width=0.5, head_length=0.5, color='black',label='eigen Vectors')
plt.ylim(-5,5)
plt.xlim(-5,5)

ax.arrow(o11,o22, V1[0,1],V1[1,1] , head_width=0.5, head_length=0.5, color='black')

plt.scatter(x1[0,:],x1[1,:],marker='^',label='scatter plot of 2nd disctribution')
plt.legend()  
plt.xlabel('X1')
plt.ylabel('X2')   
plt.title('Eigen vectors along with distribution 2 ')
plt.show()



# In[ ]:




