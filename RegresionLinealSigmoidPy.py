#!/usr/bin/env python
# coding: utf-8

# # Regresión lineal 
# https://amitnikhade.com/2021/11/18/linear-regression/
# Si dividimos la palabra regresión lineal. Aquí la palabra "lineal", solo considérala como una línea y la regresión se puede definir como la relación lineal entre la variable dependiente e independiente. Se puede representar por Y=mx+c
# Donde Y es la variable dependiente, que tiene que ser predicha (también conocida como respuesta, observación).. m es la pendiente, es decir, la pendiente en la línea también conocida como gradiente o coeficiente. x es la variable independiente (también conocida como predictor, variable exploratoria) que son los datos, y c es la intersección, es decir, el extremo inicial que cruza el eje y.
# La regresión lineal se puede clasificar además como regresión lineal simple y regresión lineal múltiple. La regresión lineal simple implica una sola variable, mientras que la regresión lineal múltiple implica múltiples variables.
# La relación entre las variables independientes y dependientes se observa mediante correlación.
# Residual es el término de error que se obtiene restando los valores predichos de los valores reales.
# Las predicciones de la regresión lineal son siempre continuas.
# Si el valor del eje Y es constante, entonces la pendiente formada será negativa ( y = -mx + c )..  

# Regresion logistic
# https://amitnikhade.medium.com/logistic-regression-lucid-explanation-8e79db05350f
# https://amitnikhade.com/
# ![image.png](attachment:image.png)
# ![image-2.png](attachment:image-2.png)

# In[95]:


# import packages
import numpy as np
from numpy.random import randn, randint
from numpy.random import seed
from matplotlib import pyplot as plt


# In[96]:


#initialize label and target
x=np.random.randn(100)
y=np.random.randn(100)+x


# In[97]:


#visualize the data
plt.scatter(x,y, c = 'blue',cmap='viridis')
plt.grid()
plt.show()


# In[98]:


c, m = 0.0,1.0
lr = 0.000001
epochs = 40
error = []
epoch_cost = []
predictions = []
for epoch in range(epochs):
    cost_c, cost_m = 0, 0
    for i in range(len(x)):
        #Making predictions        
        y_pred = (c + m*x[i])
        predictions.append(y_pred)
        print('epoch:',epoch ,'cost:',((y[i] - y_pred)**2)/len(y))
        #Gradient Descent
        for j in range(len(x)):
            partial_derivative_c = -2 * (y[j] - (c + m*x[j]))
            partial_derivative_m = (-2 * x[j]) * (y[j] - (c + m*x[j]))
            cost_c += partial_derivative_c
            cost_m += partial_derivative_m
        #Updating coeff.
        c = c - lr * cost_c
        m = m - lr * cost_m
    error.append(epoch_cost)
#plotting the prediction
pred = predictions[-100:]
plt.scatter(x,pred, c = 'red',cmap='viridis')
plt.scatter(x,y, c = 'blue',cmap='viridis')
plt.grid()
plt.show()


# # Regresion logistica o Sigmoide
# https://amitnikhade.com/2021/12/24/logistic-regression-lucid-explanation/
# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# Sigmoide
# Sigmoid es una curva en forma de S que transforma los valores del rango 0 a 1, también conocido como función logística. Como sabemos, estamos discutiendo el algoritmo de clasificación, por lo que esperamos una salida como un valor discreto, ya sea 1 o 0. pero en el caso de la regresión lineal cuando intentamos separar los puntos de datos con la recta obtenemos valores continuos que pueden estar por debajo de cero así como por encima de 1 o valores flotantes, eso no es lo que necesitamos. Así que para superar este problema y obtener una salida clara en forma binaria, necesitamos la función sigmoide que escala los valores entre 0 y 1. Si el valor es mayor que 0,5 lo consideramos como uno y si es menor que 0,5 lo consideramos como 0. Y por lo tanto obtenemos una salida cristalina.
# ![image.png](attachment:image.png)
# Función sigmoide o logística. La formulación de la función sigmoide se da como
# ![image-2.png](attachment:image-2.png)
# Nota: e es el número de Euler. x es el valor pasado a la ecuación.
# 
# En Python, el sigmoide está codificado como
# 
#                       sigmoid = 1 / (1 + e**(-z))

# In[99]:


import numpy as np
from numpy.random import randn, randint
from numpy.random import seed
from matplotlib import pyplot as plt
import pandas as pd


# In[100]:


df = pd.DataFrame({'infect_rate':[-4,-8,-2,-9,-5,-1,41,-18,-12,-29,-25,-12,42,38,-23,39,53,-31,44,84,-24,94,55,-15], 'infected/not':[0,1,0,1,1,0,0,1,0,1,1,0,0,1,0,1,1,0,0,1,0,1,1,0]})
#df


# In[101]:


from numpy import log, dot, e
from numpy.random import rand

class LogisticRegression:
    
    def sigmoid(self, z): 

        return 1/(1+np.exp(-z))
    
    def cost_function(self, X, y, weights):                 
        z = dot(X, weights)
        predict_1 = y * log(self.sigmoid(z))
        predict_0 = (1 - y) * log(1 - self.sigmoid(z))
        return -sum(predict_1 + predict_0) / len(X)
    
    def fit(self, X, y, epochs=10, lr=0.001):        
        loss = []
        weights = rand(X.shape[1])

        
        N = len(X)

                 
        for _ in range(epochs):   
            y_hat = self.sigmoid(dot(X, weights))

            weights -= lr * dot(X.T,  y_hat - y) / N            

            loss.append(self.cost_function(X, y, weights)) 
            print(_)

            plt.scatter(X,y_hat, c = 'red',cmap='viridis')
            plt.grid()
            plt.show()
        self.weights = weights
        self.loss = loss
    
    def predict(self, X):        
        z = dot(X, self.weights)
        return [1 if i > 0.5 else 0 for i in self.sigmoid(z)]


# In[102]:


x = np.array(df.infect_rate).reshape(24,1)
y = np.array(df['infected/not']).reshape(24)


# In[103]:


plt.scatter(x,y, c = 'red',cmap='viridis')
plt.grid()
plt.show()


# In[104]:


L = LogisticRegression()
L.fit(x,y)


# In[105]:


test = np.array([39]).reshape(1,1)
L.predict(test)


# # Referencias
# https://arxiv.org/
# https://amitnikhade.com/2021/12/24/logistic-regression-lucid-explanation/

# In[ ]:




