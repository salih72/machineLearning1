import pandas as pd
import numpy as np
import os
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, render_template

main = Flask(__name__)


data = pd.read_csv("/Users/salihcsr/Documents/GitHub/ML_Example/breast-cancer.csv")


data.info()
print("----------------")
print(data.head(5))
print("----------------")
data = data.drop(columns=["id"])
print(data.head(5))
print("----------------")


print(data.loc[30])
print("----------------")


Diagnosis_num = np.zeros((data.shape[0], 1)).astype(int)
for i in range(data.shape[0]):
    if data['diagnosis'][i] == 'M':
        Diagnosis_num[i] = 1

Diagnosis_num = pd.DataFrame(Diagnosis_num, columns=["Diagnosis_num"])
data.drop(['diagnosis'], axis=1, inplace=True)
data = pd.concat([Diagnosis_num, data], axis=1)

print("-----------------")
print(data.head(5))
print("----------------")


data_np = np.array(data)


random_siralama = np.random.permutation(data.shape[0])
data_np = data_np[random_siralama, :]


egitim_X = data_np[:int(data.shape[0] * 0.5), 1:]
egitim_y = data_np[:int(data.shape[0] * 0.5), 0]

val_X = data_np[int(data.shape[0] * 0.5):int(data.shape[0] * 0.7), 1:]
val_y = data_np[int(data.shape[0] * 0.5):int(data.shape[0] * 0.7), 0]

test_X = data_np[int(data.shape[0] * 0.7):, 1:]
test_y = data_np[int(data.shape[0] * 0.7):, 0]

print(egitim_X.shape, egitim_y.shape, val_X.shape, val_y.shape)
print("********************")

df = pd.DataFrame(data)
print(df.dtypes)
print("********************")


scaler = MinMaxScaler()
egitim_X = scaler.fit_transform(egitim_X)
val_X = scaler.transform(val_X)
test_X = scaler.transform(test_X)

def uzaklik_hesapla(ornek, matris):
    uzakliklar = np.zeros(matris.shape[0])
    for i in range(matris.shape[0]):
        uzakliklar[i] = np.sqrt(np.sum((ornek - matris[i, :]) ** 2))
    return uzakliklar

def basari_hesapla(tahmin, gercek):
    t = 0
    for i in range(len(tahmin)):
        if tahmin[i] == gercek[i]:
            t += 1
    return (t / len(tahmin)) * 100

def knn_predict(egitim_X, egitim_y, test_X, k):
    tahminler = np.zeros(test_X.shape[0])
    for i in range(test_X.shape[0]):
        ornek = test_X[i, :]
        uzakliklar = uzaklik_hesapla(ornek, egitim_X)
        yakindan_uzaga_indisler = np.argsort(uzakliklar)
        mode_result = stats.mode(egitim_y[yakindan_uzaga_indisler[:k]])
        mode_value = mode_result.mode[0] if isinstance(mode_result.mode, np.ndarray) else mode_result.mode
        tahminler[i] = mode_value
    return tahminler

aday_k_lar = [1, 3, 5, 7, 9]
best_k = None
best_accuracy = 0
validation_accuracies = []

for k in aday_k_lar:
    tahminler = knn_predict(egitim_X, egitim_y, val_X, k)
    basari = basari_hesapla(tahminler, val_y)
    print(f'k= {k} için doğrulama başarısı: {basari}')
    validation_accuracies.append(basari)
    if basari > best_accuracy:
        best_accuracy = basari
        best_k = k

print(f'En iyi k: {best_k} ile doğrulama başarısı: {best_accuracy}')


@main.route('/')
def home():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)
    final_features = scaler.transform(final_features)
    
    tahminler = knn_predict(egitim_X, egitim_y, final_features, best_k)
    mode_value = tahminler[0]
    if mode_value == 1:
        result = "Malignant"
    else:
        result = "Benign"

    return render_template('index.html', prediction_text=f'Cancer Type Prediction: {result}')

plt.figure(figsize=(14, 6))  

plt.subplot(1, 2, 1)
plt.suptitle("Graphs")
sns.barplot(x=aday_k_lar, y=validation_accuracies, palette="viridis")
plt.title('Doğrulama Başarısı vs. K Değeri (Barplot)')
plt.xlabel('K Değeri')
plt.ylabel('Doğrulama Başarısı (%)')
plt.grid(True)
plt.ylim(80, 100)


plt.subplot(1, 2, 2)
plt.plot(aday_k_lar, validation_accuracies, marker='o')
plt.title('Doğrulama Başarısı vs. K Değeri (Line Plot)')
plt.xlabel('K Değeri')
plt.ylabel('Doğrulama Başarısı (%)')
plt.grid(True)
plt.xticks(aday_k_lar)
plt.tight_layout() 
plt.pause(5)

plt.show()

if __name__ == "__main__":
    main.run(debug=True, port= 5001)











