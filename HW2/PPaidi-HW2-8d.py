#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df_train = pd.read_table("crime-train.txt")
df_test = pd.read_table("crime-test.txt")
n=1595
d=96
y=np.zeros(n)
X=df_train.drop('ViolentCrimesPerPop',axis=1).values
y=df_train['ViolentCrimesPerPop'].values
W=np.zeros(d-1)
count1=[]
lambda1=[]
sum=0;

for i in range(1595):
    sum=sum+y[i]
sum=sum/1595

lambdamax=30
ak=np.zeros(d-1)
b=0
ck=np.zeros(d-1)
check=np.zeros(d-1)
epsilon=np.zeros(d-1)
count=0
converged=False

#looping for convergence
while not converged:
    b=np.sum(y-np.dot(X,W.T))
    b=b/n
    check[:]=W
    for k in range(d-1):
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

    for i in range(d-1):
        if epsilon[i]>0.001:
            converged=False
            break 


#maximum and minimum
maxi=np.argmax(W)
mini=np.argmin(W)


print('This had the largest Lasso Coefficient')
print(df_train.iloc[:,maxi+1].name)
print('This had the Smallest Lasso Coefficient')
print(df_train.iloc[:,mini+1].name)
    


# In[ ]:




