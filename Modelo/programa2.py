import joblib
from sklearn.preprocessing import LabelEncoder

# Cargar el modelo guardado
model = joblib.load('modelo_hu.pkl')

# Definir las etiquetas originales
y_train_original = ['carro', 'martillo', 'coca', 'cerveza', 'caja', 'semaforo']

# Crear un nuevo LabelEncoder y ajustarlo a las etiquetas de entrenamiento
label_encoder = LabelEncoder()
label_encoder.fit(y_train_original)

# Nuevos momentos de Hu
new_hu_moments = [[0.15, 0.45, 0.89, 0.12, 0.34, 0.67, 0.90]]

# Hacer una predicción
predicted_label = model.predict(new_hu_moments)
predicted_label = label_encoder.inverse_transform(predicted_label)

print(f'La etiqueta predicha es: {predicted_label[0]}')
