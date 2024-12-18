import cv2
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Cargar el modelo guardado
model = joblib.load('modelo_hu.pkl')

# Definir las etiquetas originales
y_train_original = ['carro', 'martillo', 'coca', 'cerveza', 'caja', 'semaforo']

# Crear un nuevo LabelEncoder y ajustarlo a las etiquetas de entrenamiento
label_encoder = LabelEncoder()
label_encoder.fit(y_train_original)

# Función para calcular los momentos de Hu a partir de una imagen
def get_hu_moments(image_path):
    # Cargar la imagen
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Cargar en escala de grises
    
    # Asegurarse de que la imagen se cargó correctamente
    if image is None:
        raise ValueError("No se pudo cargar la imagen. Verifica la ruta.")
    
    # Aplicar umbral para convertir la imagen en binaria (opcional, depende del tipo de imágenes)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Calcular los momentos de Hu
    moments = cv2.moments(binary_image)
    hu_moments = cv2.HuMoments(moments).flatten()  # Aplanar los momentos para devolver un arreglo unidimensional
    return hu_moments

# Ruta de la imagen
image_path = 'image_166.png'

# Obtener los momentos de Hu de la nueva imagen
new_hu_moments = get_hu_moments(image_path)

# Hacer la predicción
new_hu_moments = new_hu_moments.reshape(1, -1)  # Asegurarse de que sea un arreglo 2D
predicted_label = model.predict(new_hu_moments)
predicted_label = label_encoder.inverse_transform(predicted_label)

print(f'La etiqueta predicha es: {predicted_label[0]}')
