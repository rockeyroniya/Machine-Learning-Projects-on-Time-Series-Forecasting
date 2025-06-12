#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
import pyflux as pf
from datetime import datetime
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


dataset_root = '/home/roniya/git-new/Time Series with LSTM in Machine Learning/'
data_a = pd.read_csv(os.path.join(dataset_root, 'cpu-full-a.csv'), parse_dates=[0], infer_datetime_format=True)
data_train_a = pd.read_csv(os.path.join(dataset_root, 'cpu-train-a.csv'), parse_dates=[0], infer_datetime_format=True)
data_test_a = pd.read_csv(os.path.join(dataset_root, 'cpu-test-a.csv'), parse_dates=[0], infer_datetime_format=True)
data_b = pd.read_csv(os.path.join(dataset_root, 'cpu-full-b.csv'), parse_dates=[0], infer_datetime_format=True)
data_train_b = pd.read_csv(os.path.join(dataset_root, 'cpu-train-b.csv'), parse_dates=[0], infer_datetime_format=True)
data_test_b = pd.read_csv(os.path.join(dataset_root, 'cpu-test-b.csv'), parse_dates=[0], infer_datetime_format=True)
plt.figure(figsize=(20,8))
plt.plot(data_train_a['datetime'], data_train_a['cpu'], color='black')
plt.ylabel('CPU %')
plt.title('CPU Utilization')


# In[ ]:


plt.figure(figsize=(20,8))
plt.plot(data_a['datetime'], data_a['cpu'], color='black')
plt.ylabel('CPU %')
plt.title('CPU Utilization')
plt.axvspan(xmin=pd.Timestamp(datetime(2017,1,28,1,42)), xmax=pd.Timestamp(datetime(2017,1,28,2,41)), color='#d4d4d4')


# In[ ]:


plt.figure(figsize=(20,8))
plt.plot(data_train_b['datetime'], data_train_b['cpu'], color='black')
plt.ylabel('CPU %')
plt.title('CPU Utilization')


# In[ ]:


plt.figure(figsize=(20,8))
plt.plot(data_b['datetime'], data_b['cpu'], color='black')
plt.ylabel('CPU %')
plt.title('CPU Utilization')
plt.axvspan(xmin=pd.Timestamp(datetime(2017,1,28,4,42)), xmax=pd.Timestamp(datetime(2017,1,28,5,41)), color='#d4d4d4')


# In[ ]:


model_a = pf.ARIMA(data=data_train_a, ar=11, ma=11, integ=0, target='cpu')
x = model_a.fit("M-H")
model_a.plot_fit(figsize=(20,8))
model_a.plot_predict(h=60,past_values=100,figsize=(20,8))


# In[ ]:


model_a.plot_predict_is(h=60, figsize=(20,8))


# In[ ]:


model_b = pf.ARIMA(data=data_train_b, ar=11, ma=11, integ=0, target='cpu')
x = model_b.fit("M-H")
model_b.plot_predict(h=60,past_values=100,figsize=(20,8))


# In[ ]:





# In[ ]:




