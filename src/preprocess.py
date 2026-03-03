# preprocess.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def clean_data(filepath, scale_numeric=True, save_path=None):
    """
    Cleans and preprocesses housing dataset.

    Parameters
    ----------
    filepath : str
        Path to Housing.csv
    scale_numeric : bool
        Whether to scale numeric features
    save_path : str or None
        Path to save preprocessed CSV file

    Returns
    -------
    df : pd.DataFrame
        Fully cleaned dataframe
    """


    # Load data
    df = pd.read_csv(filepath)


    # Remove duplicates
    df = df.drop_duplicates()

    # Handle binary categorical variables
    binary_cols = [
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "prefarea"
    ]

    for col in binary_cols:
        df[col] = df[col].map({"yes": 1, "no": 0})


    # Encode furnishingstatus
    le = LabelEncoder()
    df["furnishingstatus"] = le.fit_transform(df["furnishingstatus"])


    # Feature Scaling (exclude target)
    if scale_numeric:
        scaler = StandardScaler()

        scale_cols = [
            "area",
            "bedrooms",
            "bathrooms",
            "parking"
        ]

        df[scale_cols] = scaler.fit_transform(df[scale_cols])


    # Save processed dataset
    if save_path is not None:
        df.to_csv(save_path, index=False)
        print(f"Preprocessed data saved to {save_path}")

    return df