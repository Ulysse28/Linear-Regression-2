#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import SGDRegressor


# In[6]:


np.random.seed(0)
x,y = make_regression(n_samples = 100, n_features = 1, noise=10)
plt.scatter(x,y)


# In[17]:


model  = SGDRegressor(max_iter=1000, eta0=0.001)
model.fit(x,y)


# In[15]:


print('Coeff R2 = ', model.score(x,y))
plt.scatter(x,y)
plt.plot(x, model.predict(x), c='red', lw = 3)

