import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Cargar datos
data = pd.read_csv('base_datos_balanceada_datos.csv')

# Separar características (momentos de Hu) y etiquetas
X = data.iloc[:, :-1].values  # Todos los momentos de Hu (columnas menos la última)
y = data.iloc[:, -1].values   # La última columna es la etiqueta

# Codificar las etiquetas si son categóricas
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Dividir los datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Crear el modelo de clasificación
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)
# Hacer predicciones en los datos de prueba
y_pred = model.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')

import joblib

# Guardar el modelo entrenado
joblib.dump(model, 'modelo_hu.pkl')


