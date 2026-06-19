import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv("car data.csv")

# Encode text columns
encoder = LabelEncoder()

data["Car_Name"] = encoder.fit_transform(data["Car_Name"])
data["Fuel_Type"] = encoder.fit_transform(data["Fuel_Type"])
data["Selling_type"] = encoder.fit_transform(data["Selling_type"])
data["Transmission"] = encoder.fit_transform(data["Transmission"])

# Features and target
X = data.drop("Selling_Price", axis=1)
y = data["Selling_Price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy (R2 Score)
score = r2_score(y_test, y_pred)
print("R2 Score:", score)

# Example prediction
sample = X.iloc[[0]]
predicted_price = model.predict(sample)

print("Actual Price:", y.iloc[0])
print("Predicted Price:", predicted_price[0])