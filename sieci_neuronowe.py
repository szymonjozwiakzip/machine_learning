import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import random

file_path = 'archive/eurocup.csv' 
df = pd.read_csv(file_path, encoding='ISO-8859-1')

final_scores = df[(df['POINTS_A'].notna()) & (df['POINTS_B'].notna())]
final_scores = final_scores[(final_scores['MINUTE'] == 40) | (final_scores['MARKERTIME'] == '00:00')]

final_scores['TOTAL_POINTS'] = final_scores['POINTS_A'] + final_scores['POINTS_B']

final_scores = final_scores.loc[final_scores.groupby(['year', 'gamenumber'])['TOTAL_POINTS'].idxmax()]

conditions = [
    final_scores['POINTS_A'] > final_scores['POINTS_B'],
    final_scores['POINTS_A'] < final_scores['POINTS_B'],
    final_scores['POINTS_A'] == final_scores['POINTS_B']
]
choices = ['win', 'loss', 'draw']
final_scores['RESULT'] = np.select(conditions, choices)

features = ['POINTS_A', 'POINTS_B']
X = final_scores[features]
le = LabelEncoder()
y = le.fit_transform(final_scores['RESULT'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

test_indices = X_test.index

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

class BasketballNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BasketballNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_size = X_train.shape[1]
hidden_size = 64
num_classes = len(np.unique(y))
model = BasketballNN(input_size, hidden_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    y_pred = model(X_test)
    _, predicted = torch.max(y_pred, 1)
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    print(f'Accuracy: {accuracy:.4f}')

def test_predictions(model, X_test, y_test, le, df, test_indices, num_examples=10):
    model.eval()
    random_indices = random.sample(range(len(test_indices)), min(num_examples, len(test_indices)))
    with torch.no_grad():
        y_pred = model(X_test)
        _, predicted = torch.max(y_pred, 1)
        print("\nSprawdzenie predykcji dla losowych meczów:")
        for i, rand_idx in enumerate(random_indices):
            idx = test_indices[rand_idx]
            true_label = le.inverse_transform([y_test[rand_idx].item()])[0]
            predicted_label = le.inverse_transform([predicted[rand_idx].item()])[0]
            team_a = df.loc[idx, 'TeamA']
            team_b = df.loc[idx, 'TeamB']
            game_id = df.loc[idx, 'gamenumber']
            year = df.loc[idx, 'year']
            points_a = df.loc[idx, 'POINTS_A']
            points_b = df.loc[idx, 'POINTS_B']
            print(f"Przykład {i+1}: Mecz: {team_a} vs {team_b}, Rok: {year}, Game ID: {game_id}, Wynik: {points_a}-{points_b}, Prawdziwy wynik: {true_label}, Przewidziany wynik: {predicted_label}")

test_predictions(model, X_test, y_test, le, final_scores, test_indices, num_examples=10)
