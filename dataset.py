import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# cargar el dataset y limpiar nombres de columnas
df = pd.read_csv("Life Expectancy Data.csv")
df.columns = df.columns.str.strip()

X_col = 'Income composition of resources'
Y_col = 'Life expectancy'

# elimina las filas con datos faltantes en las columnas X_col y Y_col
df_clean = df[[X_col, Y_col]].dropna()

# Preparar X (variable independiente) y Y (variable dependiente)
X = np.array(df_clean[X_col]).reshape(-1, 1)
Y = np.array(df_clean[Y_col]).reshape(-1, 1)

Y_log = np.log(Y)

# entrenar modelo
model = LinearRegression(fit_intercept=True)
model.fit(X, Y_log)

# Obtener los coeficientes
intercept = model.intercept_[0]
coef = model.coef_[0][0]

# rango de valores X para la curva de regresión
X_new = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

# predecir Y
Y_new = np.exp(model.predict(X_new))

# crear grafico
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X, Y, label='Datos Reales')
ax.plot(X_new, Y_new, color='red', label=r'Regresión Exponencial')

ax.set_xlabel(f'Índice Composición de ingresos de los recursos')
ax.set_ylabel('Expectativa de vida (Años)')
ax.set_title(f'Prediccion de Expectativa de Vida basada en Indice Composición de ingresos de los recursos')
ax.legend()
ax.grid(True)
plt.show()

# Imprimir resultados del modelo
print(f"\nModelo ln(Y) = a + b*X:")
print(f"Intercepto (a): {intercept:.4f}")
print(f"Coeficiente (b) para '{X_col}': {coef:.4f}")