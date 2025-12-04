# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 22-09-2025



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv("Sunspots.csv")
print("Columns in dataset:", data.columns)
print(data.head())

sunspots = data['Monthly Mean Total Sunspot Number'].dropna()

plt.figure(figsize=(10,5))
plt.plot(sunspots, label='Sunspots Data')
plt.title('Sunspot Time Series')
plt.xlabel('Time Index'); plt.ylabel('Monthly Mean Sunspot Number')
plt.legend(); plt.grid(); plt.show()

fig, ax = plt.subplots(2, 1, figsize=(8,6))
plot_acf(sunspots, lags=30, ax=ax[0])
ax[0].set_title('ACF of Sunspots')
plot_pacf(sunspots, lags=30, ax=ax[1])
ax[1].set_title('PACF of Sunspots')
plt.tight_layout(); plt.show()

ar1 = np.array([1, -0.5])
ma1 = np.array([1, 0.5])
arma1 = ArmaProcess(ar1, ma1).generate_sample(nsample=len(sunspots))

plt.figure(figsize=(10,5))
plt.plot(arma1, label='Simulated ARMA(1,1)')
plt.title('Simulated ARMA(1,1) Process')
plt.xlabel('Time Index'); plt.ylabel('Simulated Value')
plt.legend(); plt.grid(); plt.show()

fig, ax = plt.subplots(2, 1, figsize=(8,6))
plot_acf(arma1, lags=20, ax=ax[0])
ax[0].set_title('ACF of ARMA(1,1)')
plot_pacf(arma1, lags=20, ax=ax[1])
ax[1].set_title('PACF of ARMA(1,1)')
plt.tight_layout(); plt.show()

ar2 = np.array([1, -0.33, 0.5])
ma2 = np.array([1, 0.9, 0.3])
arma2 = ArmaProcess(ar2, ma2).generate_sample(nsample=len(sunspots))

plt.figure(figsize=(10,5))
plt.plot(arma2, label='Simulated ARMA(2,2)', color='orange')
plt.title('Simulated ARMA(2,2) Process')
plt.xlabel('Time Index'); plt.ylabel('Simulated Value')
plt.legend(); plt.grid(); plt.show()

fig, ax = plt.subplots(2, 1, figsize=(8,6))
plot_acf(arma2, lags=20, ax=ax[0])
ax[0].set_title('ACF of ARMA(2,2)')
plot_pacf(arma2, lags=20, ax=ax[1])
ax[1].set_title('PACF of ARMA(2,2)')
plt.tight_layout(); plt.show()
```

OUTPUT:
## SIMULATED ARMA(1,1) PROCESS:

<img width="999" height="545" alt="image" src="https://github.com/user-attachments/assets/391c626f-64de-44a9-ab1b-c9c0c193f62c" />


### Partial Autocorrelation and Autocorrelation:
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/fe7a7915-7519-4bd9-8b13-ec000a3cb2bd" />






## SIMULATED ARMA(2,2) PROCESS:
<img width="999" height="545" alt="image" src="https://github.com/user-attachments/assets/8eb68a13-037c-4c9a-bd85-850e09fea7b8" />


### Partial Autocorrelation and Autocorrelation:

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/783dd3d4-b619-4fc8-aa1d-2eb6e7a4dafd" />





RESULT:
Thus, a python program is created to fir ARMA Model successfully.
