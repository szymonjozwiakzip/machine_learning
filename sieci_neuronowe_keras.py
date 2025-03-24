import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

file_path = 'archive/eurocup.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')

final_scores = df[(df['POINTS_A'].notna()) & (df['POINTS_B'].notna())]
final_scores = final_scores[(final_scores['MINUTE'] == 40) | (final_scores['MARKERTIME'] == '00:00')]

final_scores['TOTAL_POINTS'] = final_scores['POINTS_A'] + final_scores['POINTS_B']

final_scores = final_scores.sort_values(by=['year', 'gamenumber', 'MARKERTIME'])
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

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

noise_factor = 0.1
X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)

model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu', kernel_regularizer=l2(0.02)),
    Dropout(0.5),
    Dense(32, activation='relu', kernel_regularizer=l2(0.02)),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

num_epochs = 30
history = model.fit(X_train_noisy, y_train, epochs=num_epochs, batch_size=32, verbose=1, validation_data=(X_test_noisy, y_test))

loss, accuracy = model.evaluate(X_test_noisy, y_test, verbose=0)
print(f'Accuracy: {accuracy:.4f}')

def test_predictions_keras(model, X_test, y_test, le, df, num_examples=10):
    random_indices = random.sample(range(len(X_test)), min(num_examples, len(X_test)))
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("\nSprawdzenie predykcji dla losowych meczów:")
    for i, idx in enumerate(random_indices):
        true_label = le.inverse_transform([y_test[idx]])[0]
        predicted_label = le.inverse_transform([y_pred_classes[idx]])[0]
        team_a = df.iloc[idx]['TeamA']
        team_b = df.iloc[idx]['TeamB']
        game_id = df.iloc[idx]['gamenumber']
        year = df.iloc[idx]['year']
        print(f"Przykład {i+1}: Mecz: {team_a} vs {team_b}, Rok: {year}, Game ID: {game_id}, "
              f"Prawdziwy wynik: {true_label}, Przewidziany wynik: {predicted_label}")

test_predictions_keras(model, X_test_noisy, y_test, le, final_scores, num_examples=10)
