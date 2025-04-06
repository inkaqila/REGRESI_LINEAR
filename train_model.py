import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv('dataset/advertising.csv')

# Pisahkan fitur dan target
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Buat dan latih model
model = LinearRegression()
model.fit(X, y)

# Simpan model ke file .pkl
with open('model_regresi.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model berhasil disimpan ke model_regresi.pkl")
