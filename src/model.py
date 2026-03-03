# src/model.py

import pandas as pd
from preprocess import clean_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


# 1. Load & clean data
data = clean_data(
    "../data/raw_data.csv",
    scale_numeric=True,
    save_path="../data/preprocessed_data.csv"
)


# 2. Define features
numeric_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
categorical_cols = ['mainroad','guestroom','basement','hotwaterheating',
                    'airconditioning','prefarea','furnishingstatus']

X = data.drop('price', axis=1)
y = data['price']


# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4. Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ]
)


# 5. Model Pipeline: preprocessing + model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])


# 6. Train the model
model_pipeline.fit(X_train, y_train)


# 7. Save artifacts
# Save the whole pipeline (includes preprocessing)
joblib.dump(model_pipeline, '../backend/artifacts/house_model.pkl')

# Save test set for evaluation
joblib.dump(X_test, '../backend/artifacts/X_test.pkl')
joblib.dump(y_test, '../backend/artifacts/y_test.pkl')

print("Model pipeline and test data saved in artifacts folder.")