import joblib
import pandas as pd

# Load pipeline
model_pipeline = joblib.load('../backend/artifacts/house_model.pkl')

# Example new data
new_data = pd.DataFrame([
    {"area": 7420, "bedrooms": 4, "bathrooms": 2, "stories": 3,
     "mainroad": "yes", "guestroom": "no", "basement": "no",
     "hotwaterheating": "no", "airconditioning": "yes",
     "parking": 2, "prefarea": "yes", "furnishingstatus": "furnished"}
])

# Predict directly (pipeline handles preprocessing)
prediction = model_pipeline.predict(new_data)
print(f"Predicted Price = ${prediction[0]:,.2f}")