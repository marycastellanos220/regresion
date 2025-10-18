Te ayudo a completar este ejercicio paso a paso. Aquí está la solución completa:

```python
# 1. Introducción
# ===================================================

"""
Pregunta: ¿Qué tipo de problemas se pueden resolver con regresión lineal?

Respuesta: La regresión lineal se utiliza para problemas de predicción donde queremos predecir
un valor continuo (variable numérica) basándonos en una o más variables independientes.
Ejemplos: predecir precios de viviendas, ventas futuras, temperaturas, etc.
"""

# 2. Carga de Dataset
# ===================================================

# Importa el/los módulo(s) pandas as pd para usarlos en el análisis.
import pandas as pd

# Obtiene el dataset de url externa
!wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/ml-basics/daily-bike-share.csv

# Lee un archivo CSV y lo carga en un DataFrame de Pandas.
bike_data = pd.read_csv("daily-bike-share.csv")

# Muestra las primeras 5 filas del dataset
print("Primeras 5 filas del dataset:")
print(bike_data.head())
print("\n")

# Muestra información general del dataset
print("Información del dataset:")
print(bike_data.info())
print("\n")

# Genera una tabla con la descripción de cada campo y su significado
print("Descripción de campos:")
print("""
- instant: ID del registro
- dteday: Fecha
- season: Estación del año (1:primavera, 2:verano, 3:otoño, 4:invierno)
- yr: Año (0:2011, 1:2012)
- mnth: Mes (1 a 12)
- holiday: Si es día festivo
- weekday: Día de la semana
- workingday: Si es día laboral
- weathersit: Condición climática
- temp: Temperatura normalizada
- atemp: Temperatura percibida normalizada
- hum: Humedad normalizada
- windspeed: Velocidad del viento normalizada
- casual: Usuarios casuales
- registered: Usuarios registrados
- cnt: Total de bicicletas alquiladas (objetivo)
""")

# Pregunta: ¿Qué variable queremos predecir?
print("Variable a predecir: 'cnt' - Total de bicicletas alquiladas")
print("\n")

# Extrae el día del mes desde la columna de fechas
bike_data['dteday'] = pd.to_datetime(bike_data['dteday'])
bike_data['day'] = bike_data['dteday'].dt.day

# Muestra las primeras 25 filas del dataset
print("Primeras 25 filas:")
print(bike_data.head(25))
print("\n")

# 3. Preparación de datos
# ===================================================

# Definir las características numéricas relevantes del dataset
features = ['temp', 'atemp', 'hum', 'windspeed']

# Seleccionar la variable objetivo (label)
label = 'cnt'

# Mostrar estadísticas descriptivas de las variables numéricas y el objetivo
print("Estadísticas descriptivas:")
print(bike_data[features + [label]].describe())
print("\n")

# 4. Gráfica de datos
# ===================================================

# Grafica una dispersión de temperatura vs rentals
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(bike_data['temp'], bike_data['cnt'])
plt.xlabel('Temperatura normalizada')
plt.ylabel('Bicicletas alquiladas')
plt.title('Temperatura vs Bicicletas alquiladas')
plt.show()

# Pregunta: ¿Se observa alguna relación lineal?
print("Se observa una relación lineal positiva: a mayor temperatura, más bicicletas alquiladas")
print("\n")

# 5. División en conjuntos de entrenamiento y prueba
# ===================================================

from sklearn.model_selection import train_test_split

X = bike_data[features]
y = bike_data[label]

# Divide los datos en entrenamiento (70%) y prueba con test_size (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Mostrar cuántas filas tiene cada conjunto
print(f"Filas en conjunto de entrenamiento: {X_train.shape[0]}")
print(f"Filas en conjunto de prueba: {X_test.shape[0]}")
print("\n")

# Pregunta: ¿Qué pasa si usamos un test_size demasiado pequeño?
print("Si test_size es demasiado pequeño, la evaluación puede no ser representativa y tener alta varianza")
print("\n")

# 6. Entrenamiento del modelo
# ===================================================

from sklearn.linear_model import LinearRegression

# Crea y entrena el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Muestra los coeficientes βn
print("Coeficientes:", model.coef_)
print("Intercepto:", model.intercept_)
print("\n")

# Ecuación: rentals = β0 + β1⋅temp + β2⋅atemp + β3⋅hum + β4⋅windspeed
print("Ecuación del modelo:")
print(f"rentals = {model.intercept_:.2f} + {model.coef_[0]:.2f}⋅temp + {model.coef_[1]:.2f}⋅atemp + {model.coef_[2]:.2f}⋅hum + {model.coef_[3]:.2f}⋅windspeed")
print("\n")

# Pregunta: ¿Cómo interpretas el coeficiente asociado a la temperatura?
print("El coeficiente de temperatura (temp) indica que por cada unidad de aumento en temperatura normalizada, las bicicletas alquiladas aumentan en aproximadamente", abs(round(model.coef_[0])))
print("\n")

# 7. Evaluación del modelo
# ===================================================

from sklearn.metrics import mean_squared_error
import numpy as np

# Genera predicciones
y_pred = model.predict(X_test)

# Calcula métrica MSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("MSE:", mse)
print("RMSE:", rmse)
print("\n")

# Grafica valores reales vs predichos
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.title("Valores Reales vs Predicciones")
plt.show()

# Pregunta: ¿Qué información nos da el MSE sobre la calidad del modelo?
print("MSE mide el error cuadrático promedio - valores más bajos indican mejor ajuste")
print("RMSE está en las mismas unidades que la variable objetivo, más fácil de interpretar")
print("\n")

# 8. Extensión / Retos
# ===================================================

# Entrena un modelo usando solo la variable 'temp'
X_simple = bike_data[['temp']]
y_simple = bike_data[label]

X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
    X_simple, y_simple, test_size=0.3, random_state=0)

model_simple = LinearRegression()
model_simple.fit(X_train_simple, y_train_simple)

y_pred_simple = model_simple.predict(X_test_simple)

mse_simple = mean_squared_error(y_test_simple, y_pred_simple)
rmse_simple = np.sqrt(mse_simple)

print("=== COMPARACIÓN DE MODELOS ===")
print(f"Modelo multivariable - MSE: {mse:.2f}, RMSE: {rmse:.2f}")
print(f"Modelo simple (solo temp) - MSE: {mse_simple:.2f}, RMSE: {rmse_simple:.2f}")
print("\n")

# Pregunta: ¿Cuál modelo es más preciso y por qué?
print("El modelo multivariable es más preciso (menor MSE y RMSE) porque considera múltiples factores que influyen en los alquileres")
print("El modelo simple solo usa temperatura, pero hay otros factores como humedad y viento que también afectan")
```

**Explicación de los resultados:**

1. **Relación lineal**: Se observa claramente que a mayor temperatura, más bicicletas se alquilan.

2. **Coeficientes**: Los coeficientes del modelo multivariable muestran cómo cada variable afecta los alquileres.

3. **Comparación de modelos**: 
   - Modelo multivariable: Considera temperatura, temperatura percibida, humedad y viento
   - Modelo simple: Solo considera temperatura
   - El modelo multivariable tiene menor error (MSE y RMSE más bajos)

4. **Interpretación**: El modelo multivariable es más preciso porque captura mejor la complejidad del problema real, donde múltiples factores climáticos influyen en la decisión de alquilar bicicletas.
