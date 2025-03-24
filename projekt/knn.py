import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('vehicles/vehicles.csv')

data = data.iloc[28:]

data = data[['price', 'manufacturer', 'model', 'year', 'region', 'condition']]

data.dropna(inplace=True)

data = data[(data['price'] > 500) & (data['price'] < 100000)]

def categorize_price(price):
    if price < 10000:
        return 'low'
    elif price < 20000:
        return 'medium'
    else:
        return 'high'

data['price_category'] = data['price'].apply(categorize_price)

from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for column in ['manufacturer', 'model', 'region', 'condition', 'price_category']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

X = data[['manufacturer', 'model', 'year', 'region', 'condition']]
y = data['price_category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoders['price_category'].classes_))

import os
os.makedirs('vehicles', exist_ok=True)
joblib.dump(model, 'vehicles/knn_model.pkl')
for column, le in label_encoders.items():
    joblib.dump(le, f'vehicles/label_encoder_{column}.pkl')

data.to_csv('vehicles/cleaned_craigslist_vehicles.csv', index=False)
