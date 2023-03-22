import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss

from statsmodels.tsa.arima.model import ARIMA

from statsmodels.tsa.arima_process import arma_generate_sample

#%%

df=pd.read_csv('C:/Users/Admin/Desktop/UCZELNIA/2022_2023/Time Series/Laboratory/Lab6/INVCMRMT.csv', 
               index_col='DATE', parse_dates=True) 

print(df.head(10))

#%%

plt.plot(df)

#%%

fig = plot_acf(df, lags=20, zero=False)

#%%

fig = plot_pacf(df, lags=20, zero=False)

#%%

df1 = df.diff().dropna()
print(df1.head(10))

#%%

plt.plot(df1)

#%%

fig = plot_acf(df1, lags=20, zero=False)

#%%

fig = plot_pacf(df1, lags=20, zero=False)

#%%
model1 = ARIMA(df, order=(1,0,1)).fit()
print(model1.summary())

#%%
model2 = ARIMA(df, order=(2,0,0)).fit()
print(model2.summary())

#%%
model3 = ARIMA(df, order=(3,0,1)).fit()
print(model3.summary())

#%%
model4 = ARIMA(df, order=(4,0,1)).fit()
print(model4.summary())

#%%
model5 = ARIMA(df, order=(5,0,0)).fit()
print(model5.summary())

#%%
from scipy.stats.distributions import chi2

def LLR_test1(m1,m2,DF=1):
    L1 = m1.llf
    L2 = m2.llf
    LR = 2*(L2-L1)
    p = chi2.sf(LR, DF).round(3)
    return p

#%%
print(LLR_test1(model1, model2))

#%%
print(LLR_test1(model2, model3))   

#%%
print(LLR_test1(model3, model4))

#%%
from statsmodels.tsa.ar_model import AutoReg

res1 = AutoReg(df, lags=1).fit()
