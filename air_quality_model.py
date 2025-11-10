# ============================================================
# Air Quality Prediction using Random Forest Regression
# ============================================================
# This script cleans and processes air quality data from the 
# "AirQualityUCI.csv" dataset, trains Random Forest regression 
# models to predict air pollutant concentrations, evaluates their 
# performance, and predicts future pollution levels for the next month.


# --- Step 1: Import libraries ---
import pandas as pd                    # For data manipulation and analysis
import numpy as np                     # For numerical operations
import matplotlib.pyplot as plt         # For plotting
import seaborn as sns                   # For enhanced plotting styles (optional)
from sklearn.ensemble import RandomForestRegressor  # Machine learning model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.dates as mdates       # For formatting date axes on plots


# --- Step 2: Load data ---
df = pd.read_csv("AirQualityUCI.csv")   # Load the dataset from CSV

# Select relevant columns (pollutants and environmental variables)
cols = ['Date', 'Time', 'CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 
        'NOx(GT)', 'NO2(GT)', 'T', 'RH', 'AH']
df = df[cols].copy()  # Keep only the columns of interest


# --- Step 3: Preprocess data ---
# Convert date and time columns into usable datetime formats
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce').dt.hour

# Replace invalid or missing pollutant/climate values
# The dataset uses negative numbers or -200 to indicate missing data.
for col in ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)', 'T', 'RH', 'AH']:
    df[col] = df[col].apply(lambda x: np.nan if x < 0 else x)

# Sort data by time and fill missing values with linear interpolation
df = df.sort_values(by=['Date', 'Hour']).reset_index(drop=True)
df[cols[2:]] = df[cols[2:]].interpolate(method='linear', limit_direction='both')


# --- Step 4: Add time-based features ---
# These help the model capture daily and monthly trends.
df['DayOfWeek'] = df['Date'].dt.dayofweek  # 0 = Monday, 6 = Sunday
df['Month'] = df['Date'].dt.month


# --- Step 5: Add lag features (previous hour/day pollutant levels) ---
# These allow the model to learn temporal dependencies in air quality.
pollutants = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
for pollutant in pollutants:
    df[f'{pollutant}_lag1'] = df[pollutant].shift(1)     # Value 1 hour ago
    df[f'{pollutant}_lag24'] = df[pollutant].shift(24)   # Value 24 hours ago

# Drop rows with NaN values created by lagging
df = df.dropna().reset_index(drop=True)


# --- Step 6: Train/test split ---
# We'll use the last 720 hours (~30 days) as the test (validation) set.
test_hours = 720
train = df.iloc[:-test_hours]
test = df.iloc[-test_hours:]


# --- Step 7: Define feature columns for model input ---
# Includes time, weather, and pollutant lag features
feature_cols = ['Hour', 'T', 'RH', 'AH', 'DayOfWeek', 'Month'] + \
               [f'{p}_lag1' for p in pollutants] + [f'{p}_lag24' for p in pollutants]

X_train = train[feature_cols]
X_test = test[feature_cols]

# Save the cleaned and processed dataset for future reference
df.to_csv("cleaned_airquality_data.csv", index=False)


# --- Step 8: Train Random Forest models ---
# Train one model per pollutant to predict its concentration.
results = {}
for pollutant in pollutants:
    print(f"\nTraining model for {pollutant}...")
    
    y_train = train[pollutant]
    y_test = test[pollutant]
    
    # Define and train the Random Forest model
    rf = RandomForestRegressor(
        n_estimators=500,   # Number of trees
        max_depth=15,       # Maximum depth of each tree
        random_state=42,    # Ensures reproducibility
        n_jobs=-1           # Use all CPU cores
    )
    
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    # Store model and evaluation metrics
    results[pollutant] = {
        'model': rf,
        'y_test': y_test,
        'y_pred': y_pred,
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }


# --- Step 9: Display evaluation summary ---
# Prints R², MAE, and RMSE metrics for each pollutant model
print("\n--- Model Performance Summary (Validation on March Data) ---")
for pollutant, metrics in results.items():
    print(f"{pollutant}: R²={metrics['r2']:.3f}, MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}")


# Units used for plotting labels
units = {
    'CO(GT)': 'mg/m³',
    'NMHC(GT)': 'µg/m³',
    'C6H6(GT)': 'µg/m³',
    'NOx(GT)': 'ppb',
    'NO2(GT)': 'µg/m³'
}


# --- Step 9b: Create a summary table of validation metrics ---
metrics_data = []
for pollutant, metrics in results.items():
    if pollutant != 'NMHC(GT)':  # skip NMHC(GT) if desired
        metrics_data.append([pollutant, f"{metrics['r2']:.3f}", f"{metrics['rmse']:.3f}"])

col_labels = ["Pollutant", "R² (March)", "RMSE (March)"]

fig, ax = plt.subplots(figsize=(6, 2.5))
ax.axis('off')
table = ax.table(cellText=metrics_data, colLabels=col_labels, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)

fig.suptitle("Validation Metrics for March (Actual vs Predicted)", fontsize=14)
plt.tight_layout()
plt.show()


# --- Step 10: Plot Actual vs Predicted pollutant concentrations ---
test_dates = pd.to_datetime(test['Date'])

for pollutant, metrics in results.items():
    plt.figure(figsize=(12, 5))
    plt.plot(test_dates, metrics['y_test'].values, label='Actual', linewidth=2)
    plt.plot(test_dates, metrics['y_pred'], linestyle='--', label='Predicted', color='orange')
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.title(f"{pollutant}: Actual vs Predicted Concentration (Validation - March)")
    plt.xlabel("Date")
    plt.ylabel(f"Concentration ({units[pollutant]})")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# --- Step 11: Predict next month’s (future) concentrations ---
# Generate synthetic data for the next 720 hours (≈ 30 days)
future_hours = 720
last_row = df.iloc[-1].copy()  # last observed values

# Create future datetime range (April)
future_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(hours=1), periods=future_hours, freq='H')
future_df = pd.DataFrame({
    'Date': future_dates,
    'Hour': future_dates.hour,
    'DayOfWeek': future_dates.dayofweek,
    'Month': future_dates.month,
})

# Use last known climate values for prediction baseline
for col in ['T', 'RH', 'AH']:
    future_df[col] = last_row[col]

# Initialize lag features (to be filled iteratively)
for pollutant in pollutants:
    future_df[f'{pollutant}_lag1'] = np.nan
    future_df[f'{pollutant}_lag24'] = np.nan

# Iteratively predict each hour using previous predictions as lag values
predictions = {pollutant: [] for pollutant in pollutants}
for i in range(future_hours):
    row_features = []
    for col in feature_cols:
        # Use previous predictions for lag values
        if col.endswith('_lag1'):
            pollutant = col.replace('_lag1', '')
            row_features.append(df[pollutant].iloc[-1] if i == 0 else predictions[pollutant][-1])
        elif col.endswith('_lag24'):
            pollutant = col.replace('_lag24', '')
            if i < 24:
                row_features.append(df[pollutant].iloc[-24 + i])
            else:
                row_features.append(predictions[pollutant][-24])
        else:
            row_features.append(future_df.iloc[i][col])
    
    # Predict each pollutant for this time step
    for pollutant in pollutants:
        model = results[pollutant]['model']
        pred = model.predict([row_features])[0]
        predictions[pollutant].append(pred)


# --- Step 12: Plot future predictions ---
for pollutant in pollutants:
    plt.figure(figsize=(12, 5))
    plt.plot(future_dates, predictions[pollutant], label='Predicted Future', color='purple')
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.title(f"{pollutant}: Predicted Concentration for Next Month (April)")
    plt.xlabel("Date")
    plt.ylabel(f"Concentration ({units[pollutant]})")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# --- Step 12b: Save predictions to file ---
# Combine future time data and model predictions into one DataFrame
future_predictions_df = future_df.copy()
for pollutant in pollutants:
    future_predictions_df[pollutant] = predictions[pollutant]

# Save to CSV for later analysis
future_predictions_df.to_csv("predicted_airquality_next_month.csv", index=False)
print("Future predictions saved to predicted_airquality_next_month.csv")
