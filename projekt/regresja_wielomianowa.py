import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data = pd.read_csv('vehicles/vehicles.csv')

data = data.iloc[28:]

data = data[['price', 'manufacturer', 'model', 'year', 'region', 'condition']]

data.dropna(inplace=True)

data = data[(data['price'] > 500) & (data['price'] < 100000)]

from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for column in ['manufacturer', 'model', 'region', 'condition']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

X = data[['manufacturer', 'model', 'year', 'region', 'condition']]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=5)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

import os
os.makedirs('vehicles', exist_ok=True)
joblib.dump(model, 'vehicles/polynomial_regression_model.pkl')
joblib.dump(poly, 'vehicles/polynomial_features.pkl')
for column, le in label_encoders.items():
    joblib.dump(le, f'vehicles/label_encoder_{column}.pkl')

data.to_csv('vehicles/cleaned_craigslist_vehicles.csv', index=False)
