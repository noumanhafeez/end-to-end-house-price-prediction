import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

# Load model and test data
model = joblib.load('../backend/artifacts/house_model.pkl')
X_test = joblib.load('../backend/artifacts/X_test.pkl')
y_test = joblib.load('../backend/artifacts/y_test.pkl')

# Make predictions
y_pred = model.predict(X_test)

# Plot Predicted vs Actual
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # perfect prediction line
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predicted vs Actual House Prices")
plt.grid(True)
plt.show()
plt.savefig('../backend/artifacts/predicted_vs_actual.png', dpi=300)


# Optional: print R² score on plot
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.2f}")