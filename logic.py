import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

#-------------------------------------------Definimos los datos para el entrenamiento-------------------------------------------------------------------------------------------------------------------------------#
# Datos ficticios del sistema de transporte
data = {
    'origen': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'E'],
    'destino': ['B', 'C', 'C', 'D', 'D', 'E', 'E', 'F'],
    'tiempo_viaje': [10, 15, 5, 10, 20, 10, 5, 15]
}

# Crear un DataFrame
df = pd.DataFrame(data)

# Preprocesamiento de los datos
all_labels = pd.concat([df['origen'], df['destino']]).unique()
le = LabelEncoder()
le.fit(all_labels)

df['origen_encoded'] = le.transform(df['origen'])
df['destino_encoded'] = le.transform(df['destino'])

# Normalizar de manera estadistica los tiempos de viaje para que resulte mas facil al modelo aprender
scaler = MinMaxScaler()
df['tiempo_normalizado'] = scaler.fit_transform(df[['tiempo_viaje']])

# Definir las carateristica
X = df[['origen_encoded', 'destino_encoded']].values
y = df['tiempo_normalizado'].values





#-------------------------------------------Terminamos entrenamiento-------------------------------------------------------------------------------------------------------------------------------#




#----------------------------------------------------------------------empezamos a definir, compilar el modelo y comenzar el entrenamiento-----------------------------------------------------------------------------------------------------------#

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Construimos la neurona

model = Sequential([
    Embedding(input_dim=len(le.classes_), output_dim=4, input_length=2),
    Flatten(),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Entrenamos el modelo
history = model.fit(X_train, y_train, epochs=50, batch_size=4, validation_data=(X_test, y_test), verbose=1)

# Evaluar el modelo
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Mean Absolute Error en el conjunto de prueba: {mae:.4f}")

#----------------------------------------------------------------------------------finalizacion de la compilacion e entrenamiento-----------------------------------------------------------------------------------------------#



#----------------------------------colocamos al modelo a predecir-------------------------------------------------------------------------------------------------------------------#
# Predecir el tiempo entre dos estaciones
origen = 'A'
destino = 'E'

# Convertir estaciones a valores numeros codificados que el modelo pueda entender
origen_encoded = le.transform([origen])[0]
destino_encoded = le.transform([destino])[0]


input_data = np.array([[origen_encoded, destino_encoded]])

# Predecir tiempo
predicted_time = model.predict(input_data)  # Forma [[origen, destino]]

# Desnormalizar el tiempo predicho para obtener el valor real en minutos
predicted_time_real = scaler.inverse_transform(predicted_time.reshape(-1, 1))[0][0]

print(f"Tiempo estimado de viaje entre {origen} y {destino}: {predicted_time_real:.2f} minutos")

#----------------------------------finalizacion de la prediccion-------------------------------------------------------------------------------------------------------------------#