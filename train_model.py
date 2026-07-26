# Import Libraries
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# Load Dataset
df = pd.read_csv("data/predictive_maintenance.csv")


# Feature Selection
#X = df.iloc[:, 2:8]
#y = df.iloc[:, -1]
features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[features]
y = df["Failure Type"]


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Encode Type (L → 0, M → 1, H → 2)
type_encoder = OrdinalEncoder(categories=[["L", "M", "H"]])

X_train[["Type"]] = type_encoder.fit_transform(X_train[["Type"]])
X_test[["Type"]] = type_encoder.transform(X_test[["Type"]])

# Encode Target
label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)


# Train Model
random_forest = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

random_forest.fit(X_train, y_train)


# Create models folder
os.makedirs("models", exist_ok=True)

# Save Model & Encoder
joblib.dump(random_forest, "models/model.joblib")
joblib.dump(type_encoder, "models/type_encoder.joblib")
joblib.dump(label_encoder, "models/label_encoder.joblib")

print("\n[SUCCESS] Model & Encoder training complete. Files saved successfully.")

