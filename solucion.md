

"""
Pregunta: ¿Qué tipo de problemas se pueden resolver con regresión lineal?

Respuesta: La regresión lineal se utiliza para problemas de predicción donde queremos predecir
un valor continuo (variable numérica) basándonos en una o más variables independientes.
Ejemplos: predecir precios de viviendas, ventas futuras, temperaturas, etc.


**Explicación de los resultados:**

1. **Relación lineal**: Se observa claramente que a mayor temperatura, más bicicletas se alquilan.

2. **Coeficientes**: Los coeficientes del modelo multivariable muestran cómo cada variable afecta los alquileres.

3. **Comparación de modelos**: 
   - Modelo multivariable: Considera temperatura, temperatura percibida, humedad y viento
   - Modelo simple: Solo considera temperatura
   - El modelo multivariable tiene menor error (MSE y RMSE más bajos)

4. **Interpretación**: El modelo multivariable es más preciso porque captura mejor la complejidad del problema real, donde múltiples factores climáticos influyen en la decisión de alquilar bicicletas.
