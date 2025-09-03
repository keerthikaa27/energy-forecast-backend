import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

df = pd.read_csv("energy_usage.csv")

train_values = df["usage"].values

scaler = MinMaxScaler()
scaler.fit(train_values.reshape(-1, 1))

joblib.dump(scaler, "scaler_multi.pkl")
print("Scaler retrained using CSV data and saved.")
