# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 

### NAME:RAGUNATH R
### REGISTER NO: 212222240081

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load the dataset with appropriate parsing
file_path = '/content/daily-minimum-temperatures-in-me.csv'
data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
data.columns = ['Temperature']  # Rename for simplicity

# Convert the 'Temperature' column to numeric, handling any non-numeric entries
data['Temperature'] = pd.to_numeric(data['Temperature'], errors='coerce')
data = data.dropna()  # Drop rows with NaN values

# Step 1: Data Preprocessing
print(data.head())
print(data.describe())

# Step 2: Check Stationarity
def check_stationarity(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] <= 0.05:
        print("The series is stationary.")
    else:
        print("The series is not stationary. Differencing may be needed.")

check_stationarity(data['Temperature'])

# Step 3: Differencing if needed
data['Temperature_diff'] = data['Temperature'].diff().dropna()
check_stationarity(data['Temperature_diff'].dropna())

# Step 4: Plot ACF and PACF to determine p, d, q parameters
plt.figure(figsize=(12,6))
plt.subplot(121)
plot_acf(data['Temperature_diff'].dropna(), ax=plt.gca(), lags=30)
plt.subplot(122)
plot_pacf(data['Temperature_diff'].dropna(), ax=plt.gca(), lags=30)
plt.show()

# Step 5: Fit SARIMA Model (Example parameters p=1, d=1, q=1; adjust as needed)
# Seasonal order (1,1,1,12) assuming monthly seasonality (adjust if needed)
model = SARIMAX(data['Temperature'], order=(1,1,1), seasonal_order=(1,1,1,12))
model_fit = model.fit(disp=False)

# Print model summary
print(model_fit.summary())

# Step 6: Forecasting
forecast_steps = 30
forecast = model_fit.get_forecast(steps=forecast_steps)
forecast_ci = forecast.conf_int()

# Plotting the forecast results
plt.figure(figsize=(10,5))
plt.plot(data['Temperature'], label='Observed')
plt.plot(forecast.predicted_mean, color='r', label='Forecast')
plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink')
plt.legend()
plt.show()

# Step 7: Model Evaluation with Train-Test Split
train_size = int(len(data) * 0.8)
train, test = data['Temperature'][:train_size], data['Temperature'][train_size:]

# Fit the SARIMA model on training data
model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
model_fit = model.fit(disp=False)

# Forecast on the test set
predictions = model_fit.forecast(len(test))

# Calculate Mean Squared Error
mse = mean_squared_error(test, predictions)
print(f'Mean Squared Error: {mse}')

# Plot train, test, and forecasted values
plt.figure(figsize=(12, 6))
plt.plot(train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(test.index, predictions, label='Forecast', color='red')
plt.title('Train/Test Split with Forecast')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.show()
```

### OUTPUT:
![image](https://github.com/user-attachments/assets/077403a1-7f79-4172-8a6b-7de748b8b9e9)
![image](https://github.com/user-attachments/assets/575daefd-d5ee-486b-a015-58ede5465760)
![image](https://github.com/user-attachments/assets/0d460121-6b09-49c3-abf4-7ee0ba2c98b5)
![image](https://github.com/user-attachments/assets/a04c43ab-01be-4138-a779-0fcac3a39044)
![image](https://github.com/user-attachments/assets/2157c8c1-2d4d-4d94-8b6b-5d717bee65c4)

### RESULT:
Thus the program run successfully based on the SARIMA model.
