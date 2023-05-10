#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n = 500
d=1000
k = 100
sum=0
count=0
count1=[]
lambda1=[]
FDR=[]
TPR=[]
#initializing X
X = np.random.standard_normal(size=(n,d))
noise=np.random.standard_normal(size=n)
#
Wtrue=np.zeros(d)
for j in range(1,k+1):
    if j<k+1:
        Wtrue[j-1]=j/k
    else:
        Wtrue[j-1] = 0      
y=np.dot(X,(W.T))+ noise
lambdaa= np.zeros([d])

for j in range(n):
    sum=sum+y[j]
sum=sum/n

for k in range(100):
    lam=0
    for i in range(500):
        lam=2*X[i][k]*(y[i]-sum)+lam
    lambdaa[k]=np.linalg.norm(lam)
lambdamax=np.max(lambdaa)
lambdamax=lambdamax/1.5
#looping over lambda
while lambdamax>0.01:
    ak=np.zeros(d)
    b=np.zeros(n)
    ck=np.zeros(d)
    Wtemp=np.zeros(d)
    check=np.zeros(d)
    epsilon=np.zeros(d)
    W=np.zeros(d)
    count=0
    count10=0
    e10=0
    converged=False
    #looping over for convergence
    while not converged:
        b=np.sum(y-np.dot(X,W.T))
        b=b/n
        check[:]=W
        for k in range(1000):
            Xk=X[:,k]
            Xj=np.delete(X,k, axis=1)
            Wj=np.delete(W,k,axis=0)
            ak=2*np.sum(np.square(Xk))
            ck=2*np.sum(np.dot(Xk.T,(y-(b+np.dot(Xj,Wj.T)))))  
            if ck<-lambdamax:
                W[k]=(ck+lambdamax)/ak
            elif ck >= -lambdamax and ck <= lambdamax:
                W[k]=0
            else:
                W[k]=(ck-lambdamax)/ak 
             
        epsilon=np.absolute(np.array(W) - np.array(check))
        converged=True
        
        for i in range(d):
            if epsilon[i]>0.1:
                converged=False
                break  
          
    for i in range(d):
        if W[i]!=0:
            count=count+1
            
    for i in range(100,1000):    
        if W[i]!=0:
            count10=count10+1
            
    for i in range(100):
        if W[i]!=0:
            e10=e10+1
                
    count1.append(count)

    FDR.append(count10/(count))  # appending 
    TPR.append(e10/(100))
    
    lambda1.append(lambdamax)  # appending 
    lambdamax=lambdamax/1.5
    

plt.plot(FDR,TPR,label='FDR vs TPR') 
plt.xlabel('FDR')
plt.ylabel('TPR')
plt.title("FDR vs TPR")
plt.legend()
plt.show()


# In[ ]:




