#!/usr/bin/env python
# coding: utf-8

# In[56]:


from matplotlib import pyplot as plt
import math
import numpy as np

n = 256
f_x = []
average_varaince = []
average_bias = []
x = []
z= [1,2,4,8,16,32]
error= []
y = []
total=[]


#x_values
for i in range(1,n+1):
    x.append(i/n)

#y_values and F_x values
for i in range(256):
    y.append(4*(np.sin(math.pi*X[i]))*(np.cos(6*math.pi*X[i]*X[i]))+np.random.normal(0,1))
    f_x.append(4*(np.sin(math.pi*X[i]))*(np.cos(6*math.pi*X[i]*X[i])))


for i in range(len(z)):
    m=z[i]
    σ2 = 1
    fm = []
    sum= 0  
    sum1= 0   
    c = []
    sum2=0
    sum5=0
    fbar=[]
    
    #calculating c_j
    for j in range(1,int((n/m)+1)):   
        sum=0
        for i in range(((j-1)*m+1),j*m+1):
            sum=y[i-1]+sum
        c.append(sum/m) 
        
    #calculating fhat  
    for i in range(1,n+1):
        a=0
        s=0
        for j in range(1,int((256/m)+1)):
            if x[i-1]<=(j*m)/256 and x[i-1]>(j-1)*m/256 :
                a=1
                s=c[j-1]*a
        fm.append(s)
    
    #calculating emperical error
    for i in range(n):  
        sum1=(fm[i]-f_x[i])**2 +sum1  
    error.append(sum1/n)
    
    #calculating fbar
    for j in range(1,int(n/m)+1):
        sum=0
        for i in range((j-1)*m+1,j*m+1):
            sum=f_x[i-1]+sum
        fbar.append(sum/m)
        
    #calculating average_bais
    for j in range(1, int(n/m)+1):
        for i in range((j-1)*m+1,j*m+1):
            sum5=(fbar[j-1]-f_x[i-1])**2+sum5
    average_bias.append(sum5/256) 

    #Calculating average_Variance
    average_varaince.append(σ2/m)

for i in range(len(M)):
    total.append(average_bias[i] + average_varaince[i])

plt.plot(z,error, label = "emp")
plt.plot(z,average_varaince, label="Var")
plt.plot(z,average_bias, label="bias")
plt.plot(z,total, label="Avg Error")
plt.xlabel("m values")
plt.ylabel("Error")
plt.title("m val w.r.t error")
plt.legend()
plt.show()


# In[ ]:




