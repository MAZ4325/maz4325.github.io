#!/usr/bin/env python
# coding: utf-8

# In[37]:


#Calcular el área de la función con el Método de Montecarlo.(Integrando)


# In[38]:


import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')


# In[39]:


M=10
N=10000
resul=[]
for j in range (M):
    x,Y= np.random.uniform(0, 1, size=(2, N))
    y2= np.sqrt(x)
    i = Y>(y2)
    s = Y<=(y2)
    area=(s.sum()/N)
    result=np.append(resul,area)
    a=np.mean(result)            
real=(2/3)
print("Valor del área: ",a)
print("Valor calculado: ",real)


# In[40]:


plt.figure(figsize=(10,10))  # tamaño de la figura
plt.plot(x, y2, 'r.')

plt.plot(x[i], Y[i], 'g.')
plt.plot(x[s], Y[s], 'k.')

plt.plot(0, 0, label='Función Raíz')
plt.plot(0, 0, label='Área= {:4.4f}\nReal = {:4.4f}'.format(a,real)
         , alpha=0)
plt.axis('square')
plt.legend(frameon=True, framealpha=0.9, fontsize=16)


# In[41]:


M=10
N=10000
resul=[]
for j in range (M):
    x,Y= np.random.uniform(0, 1, size=(2, N))
    y1= x**2
    y2= np.sqrt(x)
    i = Y>=(y1)
    s = Y<=(y2)
    z= i & s ==1  #Área interior
    area=(z.sum()/N)
    result=np.append(resul,area)
    a=np.mean(result)            
real=(1/3)
print("Valor del área: ",a)
print("Valor calculado: ",real)


# In[42]:


plt.figure(figsize=(10,10))  # tamaño de la figura
plt.plot(x, y1, 'b.')
plt.plot(x, y2, 'r.')

plt.plot(x[i], Y[i], 'g.')
plt.plot(x[s], Y[s], 'g.')
plt.plot(x[z], Y[z], 'k.')

plt.plot(0, 0, label='Función Cuadrado')
plt.plot(0, 0, label='Función Raíz')
plt.plot(0, 0, label='Área= {:4.4f}\nReal = {:4.4f}'.format(a,real)
         , alpha=0)
plt.axis('square')
plt.legend(frameon=True, framealpha=0.9, fontsize=16)

