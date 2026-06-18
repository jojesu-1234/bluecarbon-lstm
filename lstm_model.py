import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 1, figsize=(12,8))

wocean = pd.read_csv('서해중부_LSTM최종데이터.csv')
eocean = pd.read_csv('동해_LSTM최종데이터.csv')

""" 결측값 확인 부분
print("-서해중부-")
print("shape: ", wocean.shape)
print(wocean.head())

print("\n-동해-")
print("shape: ", eocean.shape)
print(eocean.head())

print("\n-결측값 확인-")
print("서해중부: ", wocean.isnull().sum())
print("동해: ", eocean.isnull().sum())
"""

feature_cols = ['수온(℃)표층_scaled', '염분표층_scaled', 
                '용존산소량(㎎/L)표층_scaled', '클로로필A(㎍/L)표층_scaled',
                'bluecarbon_scaled']

target_cols = ['수소이온농도표층_scaled']

def make_sequences(df, feature_cols, target_cols, timesteps = 4):
    X, y = [], []
    features = df[feature_cols].values
    targets = df[target_cols].values

    for i in range(len(df) - timesteps):
        X.append(features[i:i+timesteps])
        y.append(targets[i+timesteps])
    return np.array(X), np.array(y)

TIMESTEPS = 4

X_wocean, y_wocean = make_sequences(wocean, feature_cols, target_cols, TIMESTEPS)
X_eocean, y_eocean = make_sequences(eocean, feature_cols, target_cols, TIMESTEPS)

"""시퀀스 생성 결과
print("시퀀스 생성 결과:")
print(f"X_wocean: {X_wocean.shape}")
print(f"y_wocean: {y_wocean.shape}")
print(f"X_eocean: {X_eocean.shape}")
print(f"y_eocean: {y_eocean.shape}")
"""

split = int(len(X_wocean) * 0.8)
X_wocean_train, X_wocean_test = X_wocean[:split], X_wocean[split:]
y_wocean_train, y_wocean_test = y_wocean[:split], y_wocean[split:]

X_eocean_train, X_eocean_test = X_eocean[:split], X_eocean[split:]
y_eocean_train, y_eocean_test = y_eocean[:split], y_eocean[split:]

"""서해중부와 동해의 test/train 분리 결과
print("-test/train")
print(f"서해중부 train: {X_wocean_train}, test: {X_wocean_test}")
print(f"서해중부 train: {y_wocean_train}, teat: {y_wocean_test}")
print(f"동해 train: {X_eocean_train}, test: {X_eocean_test}")
print(f"동해 train: {y_eocean_train}, test: {y_eocean_test}")
"""

def build_model(timesteps, n_features):
    model = Sequential([
        LSTM(64, input_shape=(timesteps, n_features), return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

N_FEATURES = 5

model_wocean = build_model(TIMESTEPS, N_FEATURES)
model_eocean = build_model(TIMESTEPS, N_FEATURES)

print("-모델 구조-")
model_wocean.summary()
model_eocean.summary()

history_wocean = model_wocean.fit(
    X_wocean_train, y_wocean_train,
    epochs=100,
    batch_size=8,
    validation_data=(X_wocean_test, y_wocean_test),
    verbose=1
)

history_eocean = model_eocean.fit(
    X_eocean_train, y_eocean_train,
    epochs=100,
    batch_size=8,
    validation_data=(X_eocean_test, y_eocean_test),
    verbose=1
)

print("학습완료")
print(f"서해중부 최종 loss: {history_wocean.history['loss'][-1]:.4f}")
print(f"동해 최종 loss: {history_eocean.history['loss'][-1]:.4f}")

pred_wocean = model_wocean.predict(X_wocean_test)
pred_eocean = model_eocean.predict(X_eocean_test)

ph_scaler_wocean = MinMaxScaler()
ph_scaler_wocean.fit(wocean[['수소이온농도표층']])

ph_scaler_eocean = MinMaxScaler()
ph_scaler_eocean.fit(eocean[['수소이온농도표층']])

pred_wocean_real = ph_scaler_wocean.inverse_transform(pred_wocean)
pred_eocean_real = ph_scaler_eocean.inverse_transform(pred_eocean)

y_wocean_real = ph_scaler_wocean.inverse_transform(y_wocean_test)
y_eocean_real = ph_scaler_eocean.inverse_transform(y_eocean_test)

print("서해중부 예측 vs 실제")
for i in range(len(pred_wocean_real)):
    print(f"실제: {y_wocean_real[i][0]:.4f} 예측: {pred_wocean_real[i][0]:.4f}")

print("\n동해 예측 vs 실제")
for i in range(len(pred_eocean_real)):
    print(f"실제: {y_eocean_real[i][0]:.4f} 예측: {pred_eocean_real[i][0]:.4f}")

axes[0].plot(y_wocean_real, label='실제 pH', marker='o', color='blue')
axes[0].plot(pred_wocean_real, label='예측 pH', marker='s', linestyle='--', color='red')
axes[0].set_title('서해중부 (블루카본 O) - 실제 vs 예측 pH')
axes[0].set_ylabel('pH')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(y_eocean_real, label='실제 pH', marker='o', color='blue')
axes[1].plot(pred_eocean_real, label='예측 pH', marker='s', linestyle='--', color='red')
axes[1].set_title('동해 (블루카본 X) - 실제 vs 예측 pH')
axes[1].set_ylabel('pH')
axes[1].set_xlabel('관측 시점 (2023년 2분기 ~ 2024년 4분기)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('pH_prediction_result.png', dpi=150)
plt.show()
print("그래프 저장 완료: pH_prediction_result.png")