import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fires_thefts_df = pd.read_csv('fires_thefts.csv')

# przygotowanie danych: pozary (X) i wlamania (y)
X = fires_thefts_df.iloc[:, 0].values 
y = fires_thefts_df.iloc[:, 1].values  

# przekszt X do tablicy 2D
X = X.reshape(-1, 1)

# normalizacja danych dla lepszej wydajności (przydatna dla gradientu prostego)
X_mean = np.mean(X)
X_std = np.std(X)
y_mean = np.mean(y)
y_std = np.std(y)

X_normalized = (X - X_mean) / X_std
y_normalized = (y - y_mean) / y_std

# inicjal wart theta (parametrow modelu) i ustawienie parametrow algorytmu gradientu prostego
theta = np.zeros(2)  # pocz. wart. theta - zera
alpha = 0.007  # wsp uczenia - w tym przypakdu może być tak niski, a algorytm i tak jest szybki
epsilon = 1e-6  # kryt stopu - jak blisko zbieznosci algorytm ma sie zatrzymac
max_iterations = 1000  # maks l iteracji

# dodanie kol jedynek do X, aby uwzglednic wyraz wolny (theta_0)
X_b = np.c_[np.ones((X_normalized.shape[0], 1)), X_normalized]

# def f kosztu (do oceny, jak dobrze model dopasowuje dane)
def compute_cost(X_b, y, theta):
    m = len(y)
    predictions = X_b.dot(theta)
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost

# alg gradientu prostego do obliczania parametrow theta
def gradient_descent(X_b, y, theta, alpha, epsilon, max_iterations):
    m = len(y)
    cost_history = []
    
    for i in range(max_iterations):
        # oblicznaie gradientu
        gradients = (1 / m) * X_b.T.dot(X_b.dot(theta) - y)
        # aktualizacja parametrow theta
        theta = theta - alpha * gradients
        # oblicznaie kosztu, by monitorowac zbieznosc
        cost = compute_cost(X_b, y, theta)
        cost_history.append(cost)
        
        # spr czy zmiana w koszcie jest mniejsza niz eps (war stopu)
        if i > 0 and abs(cost_history[-2] - cost_history[-1]) < epsilon:
            break
            
    return theta, cost_history

# uruchomienie alg grad prostego
theta_optimal, cost_history = gradient_descent(X_b, y_normalized, theta, alpha, epsilon, max_iterations)

# rozw podpunktu 1 i 2
# theta_optimal to parametry modelu \( \theta_0 \) i \( \theta_1 \), obliczone za pomocą gradientu prostego

# denormalizacja parametrow theta (przeksztalc z powrotem do rzeczywistych wart)
theta_0 = y_mean - (theta_optimal[1] * (X_mean / X_std) * y_std)  # denormalizacja wyrazu wolnego
theta_1 = theta_optimal[1] * (y_std / X_std)  # denormalizacja wsp kier

# rozw podpunktu 3
# wykres zal kosztu od liczby iteracji (pokazuje, jak zmniejsza sie koszt w czasie)
plt.figure(figsize=(8, 6))
plt.plot(cost_history, label='Cost Function', color='blue')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Zbieżność funkcji kosztu')
plt.grid(True)
plt.legend()
plt.show()

# przewid l wlaman dla 50, 100 i 200 pozarow na 1000 gospodarstw domowych
fire_values = np.array([50, 100, 200]).reshape(-1, 1)
theft_predictions = theta_0 + theta_1 * fire_values


# wykr danych oraz dopasowanej linii regresji
plt.figure(figsize=(8, 6))
plt.scatter(X, y, label='Dane oryginalne', color='blue')
plt.plot(X, theta_0 + theta_1 * X, label='Linia regresji', color='red')
plt.xlabel('Pożary na tysiąc gospodarstw domowych')
plt.ylabel('Włamania na tysiąc mieszkańców')
plt.title('Regresja liniowa: Przewidywanie liczby włamań na podstawie liczby pożarów')
plt.legend()
plt.grid(True)
plt.show()

print(f"Przewidywana liczba włamań dla 50 pożarów: {theft_predictions[0][0]:.2f}")
print(f"Przewidywana liczba włamań dla 100 pożarów: {theft_predictions[1][0]:.2f}")
print(f"Przewidywana liczba włamań dla 200 pożarów: {theft_predictions[2][0]:.2f}")
