import joblib
import numpy as np

scaler = joblib.load("models/scaler.joblib")
model = joblib.load("models/champion.joblib")

tests = [
    [10, 50, 400],
    [22, 88, 920],
    [35, 100, 2000]
]

for t in tests:
    x = scaler.transform([t])
    pred = model.predict(x)
    print(t, "->", pred[0])