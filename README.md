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

# Suppress warnings
warnings.filterwarnings('ignore')

# Step 1: Load dataset
file_path = 'Sunspots.csv'
data = pd.read_csv(file_path)

# Step 2: Inspect dataset
print("Column names in dataset:", data.columns)
print(data.head())

# Step 3: Extract the sunspot numbers
sunspots = data['Monthly Mean Total Sunspot Number'].dropna()

# Step 4: Plot the sunspot time series
plt.figure(figsize=(12,6))
plt.plot(sunspots, label='Sunspots Data')
plt.title('Sunspot Time Series')
plt.xlabel('Time Index')
plt.ylabel('Monthly Mean Sunspot Number')
plt.legend()
plt.grid()
plt.show()

# Step 5: ACF and PACF of actual data
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plot_acf(sunspots, lags=30, ax=plt.gca())
plt.title('ACF of Sunspots')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')

plt.subplot(2,1,2)
plot_pacf(sunspots, lags=30, ax=plt.gca())
plt.title('PACF of Sunspots')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')

plt.tight_layout()
plt.show()

# Step 6: Simulate ARMA(1,1)
ar1 = np.array([1, -0.5])
ma1 = np.array([1, 0.5])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=len(sunspots))

plt.figure(figsize=(12,6))
plt.plot(ARMA_1, label='Simulated ARMA(1,1)')
plt.title('Simulated ARMA(1,1) Process')
plt.xlabel('Time Index')
plt.ylabel('Simulated Value')
plt.legend()
plt.grid()
plt.show()

# ACF/PACF of ARMA(1,1)
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plot_acf(ARMA_1, lags=20, ax=plt.gca())
plt.title('ACF of ARMA(1,1)')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')

plt.subplot(2,1,2)
plot_pacf(ARMA_1, lags=20, ax=plt.gca())
plt.title('PACF of ARMA(1,1)')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')

plt.tight_layout()
plt.show()

# Step 7: Simulate ARMA(2,2)
ar2 = np.array([1, -0.33, 0.5])
ma2 = np.array([1, 0.9, 0.3])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=len(sunspots))

plt.figure(figsize=(12,6))
plt.plot(ARMA_2, label='Simulated ARMA(2,2)', color='orange')
plt.title('Simulated ARMA(2,2) Process')
plt.xlabel('Time Index')
plt.ylabel('Simulated Value')
plt.legend()
plt.grid()
plt.show()

# ACF/PACF of ARMA(2,2)
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plot_acf(ARMA_2, lags=20, ax=plt.gca())
plt.title('ACF of ARMA(2,2)')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')

plt.subplot(2,1,2)
plot_pacf(ARMA_2, lags=20, ax=plt.gca())
plt.title('PACF of ARMA(2,2)')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')

plt.tight_layout()
plt.show()
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
