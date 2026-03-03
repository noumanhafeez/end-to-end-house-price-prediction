# src/evaluate_model.py
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load trained model and test data
model = joblib.load('../backend/artifacts/house_model.pkl')
X_test = joblib.load('../backend/artifacts/X_test.pkl')
y_test = joblib.load('../backend/artifacts/y_test.pkl')

# Predict on test set
y_pred = model.predict(X_test)

# Compute evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
accuracy = r2 * 100

# Beautified output
print("\n===== MODEL EVALUATION REPORT =====\n")
print(f"Model: House Price Predictor (Random Forest)")
print("-------------------------------------------")
print(f"1. Mean Absolute Error (MAE): {mae:,.2f}")
print("   -> Average absolute difference between predicted and actual price")
print(f"2. Mean Squared Error (MSE): {mse:,.2f}")
print("   -> Average squared difference (penalizes large errors)")
print(f"3. Root Mean Squared Error (RMSE): {rmse:,.2f}")
print("   -> Square root of MSE, same units as target")
print(f"4. R² Score: {r2:.2f}")
print("   -> Proportion of variance explained by the model (1 = perfect)")
print(f"5. Approximate Accuracy: {accuracy:.2f}%")
print("\n===================================\n")