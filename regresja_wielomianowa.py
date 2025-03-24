import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv('data6.tsv', sep='\t', header=None)
X = data.iloc[:, 0].values.reshape(-1, 1)
Y = data.iloc[:, 1].values

# stopnie wielomianów do regresji
degrees = [1, 2, 5]

# rysujemy punkty danych
plt.scatter(X, Y, color='black', label='Punkty danych')

# wykonujemy regresje wielomianowa dla każdego stopnia i rysujemy wynik
for degree in degrees:
    # transformacja cech dla regresji wielomianowej
    polynomial_features = PolynomialFeatures(degree=degree)
    X_poly = polynomial_features.fit_transform(X)
    
    # regresja liniowa na przeksztalconych cechach
    model = LinearRegression()
    model.fit(X_poly, Y)
    
    # generowanie przewidywan dla okreslenia krzywej regresji
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_range_poly = polynomial_features.transform(X_range)
    Y_pred = model.predict(X_range_poly)
        
    # rysowanie krzywej regresji
    plt.plot(X_range, Y_pred, label=f'Wielomian stopnia {degree}')

# wykres
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Krzywe regresji wielomianowej')
plt.legend()
plt.show()
