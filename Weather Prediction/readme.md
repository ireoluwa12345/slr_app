## Project Setup

### Environment Configuration

To get started, you'll need to set up the project environment. You can do this using conda to manage dependencies easily.

### Steps:

Create the conda environment:
`conda create --name weather_prediction python=3.9`

Activate the environment:
`conda activate weather_prediction`

Install dependencies: Install the necessary libraries for the project using the following command:
`conda install pandas numpy scikit-learn statsmodels matplotlib`

or run

`conda install pip`
`pip install -r requirements.txt`

### Data Processing

The dataset has already been pre-processed, and the cleaned data is available in the following CSV files:

hourly_cleaned_data.csv
daily_cleaned_data.csv
Each file contains relevant columns such as temperature, humidity, wind speed, and rain data for model training and prediction.

To further process the data (such as feature extraction or scaling), you can use the code provided in the scripts. Below is a basic example for scaling the features used in the Random Forest model.
`from sklearn.preprocessing import StandardScaler`

#### Define features

```python
features = ['temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
'wind_speed_10m', 'wind_gusts_10m', 'month', 'day', 'hour']
```

#### Load the dataset

`data = pd.read_csv('hourly_cleaned_data.csv')`

#### Scale the features

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(data[features])
```

### Model Training

The project uses two models: Vector AutoRegression (VAR) for multivariate regression and Random Forest Classifier for classification. Below are the instructions on how to train each model.

### VAR Model Training

The VAR model is used to predict time series data, such as temperature and humidity. Ensure that the dataset is stationary before training.
`from statsmodels.tsa.api import VAR`

#### Load and prepare the dataset

`daily_data_x = pd.read_csv("daily_cleaned_data.csv", index_col='date', parse_dates=True)`

#### Train the VAR model

```python
var_model = VAR(daily_data_x)
var_model_fitted = var_model.fit(maxlags=365)
```

#### Save the trained model

```python
import pickle
with open('var_model.pkl', 'wb') as file:
pickle.dump(var_model_fitted, file)
```

### Random Forest Classifier Training

The Random Forest Classifier is used to predict whether it will rain or not. Here’s how to train and save the model.
`from sklearn.ensemble import RandomForestClassifier`

#### Load and preprocess the dataset

```python
X_train = train_data[features]
y_train = train_data['is_rainy']
```

#### Scale the features

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

#### Train the Random Forest Classifier

```python
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)
```

#### Save the model and scaler

```python
with open('random_forest_model.pkl', 'wb') as file:
pickle.dump(clf, file)

with open('scaler.pkl', 'wb') as file:
pickle.dump(scaler, file)
```

### Making Predictions

Once the models are trained, you can use them to make predictions on the test data.

#### VAR Model Prediction

```python
import pandas as pd
from statsmodels.tsa.api import VAR
import pickle
```

#### Load the trained VAR model

```python
with open('var_model.pkl', 'rb') as file:
var_model_fitted = pickle.load(file)
```

#### Load the cleaned data

`daily_data_x = pd.read_csv("daily_cleaned_data.csv", index_col='date', parse_dates=True)`

#### Define forecast period

```python
start_date = daily_data_x.index[-1]
end_date = pd.Timestamp('2024-08-30')
forecast_days = (end_date - start_date).days
```

#### Forecast future values

```python
lag_order = var_model_fitted.k_ar
last_observations = daily_data_x.values[-lag_order:]
forecasted_values = var_model_fitted.forecast(last_observations, steps=forecast_days)
```

#### Create DataFrame for forecasted values

```python
forecast_dates = pd.date_range(start=start_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
forecast_df = pd.DataFrame(forecasted_values, index=forecast_dates, columns=daily_data_x.columns)
```

#### Save the forecast

```python
forecast_df.to_csv("var_model_forecast.csv")
print("VAR model forecast saved to 'var_model_forecast.csv'")
```

#### Random Forest Classifier Prediction

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
```

#### Load the trained Random Forest model and scaler

```python
with open('random_forest_model.pkl', 'rb') as file:
clf = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
scaler = pickle.load(file)
```

#### Load the test dataset

```python
test_data = pd.read_csv('hourly_cleaned_data.csv')
test_data['time'] = pd.to_datetime(test_data['time'])
```

#### Define features

```python
features = ['temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
'wind_speed_10m', 'wind_gusts_10m', 'month', 'day', 'hour']
```

#### Extract and scale the test features

```python
X_test = test_data[features]
X_test_scaled = scaler.transform(X_test)
```

#### Make predictions

`y_pred = clf.predict(X_test_scaled)`

#### Add predictions to the DataFrame

`test_data['predicted_rain'] = y_pred`

#### Save the predictions

```python
test_data.to_csv("random_forest_predictions.csv", index=False)
print("Random Forest predictions saved to 'random_forest_predictions.csv'")
```

### Model Evaluation

To evaluate the models, use the metrics provided in the script, such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and others for the VAR model, and the classification report and confusion matrix for the Random Forest model.
For example, to calculate the accuracy of the Random Forest Classifier:

```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")
```
