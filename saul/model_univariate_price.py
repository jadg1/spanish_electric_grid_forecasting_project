import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, mean_squared_error

# Cargar datos
df = pd.read_csv('data.csv', sep=';', parse_dates=['datetime'])
df.sort_values(by='datetime', inplace=True)
df.set_index('datetime', inplace=True)

# Diferenciar la columna 'price' para hacerla estacionaria
df['price_diff'] = df['price'].diff().dropna()

# Búsqueda en cuadrícula para ARIMA
p = range(0, 3)
d = range(0, 2)
q = range(0, 3)
best_aic = float("inf")
best_order = None
best_model = None

for i in p:
    for j in d:
        for k in q:
            try:
                model = ARIMA(df['price'], order=(i,j,k))
                model_fit = model.fit()
                aic = model_fit.aic
                if aic < best_aic:
                    best_aic = aic
                    best_order = (i, j, k)
                    best_model = model_fit
            except:
                continue

print(f"Mejor modelo ARIMA: {best_order} con AIC: {best_aic}")

# Predicciones con el mejor modelo
predictions = best_model.forecast(steps=12)

# Crear un rango de fechas para las próximas 12 horas
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date, periods=12, freq='H')

# Graficar valores ajustados y reales
last = -48  # Últimas 48 horas
plt.figure(figsize=(10, 6))
plt.plot(df.index[last:], df['price'][last:], label='Real')
plt.plot(df.index[last:], best_model.fittedvalues[last:], label='Fitted', color='orange')
plt.plot(future_dates, predictions, label='ARIMA prediction 12h aead', color='red')
plt.title(f'Fitted Values and Real Values - ARIMA{best_order}')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.savefig('precio_arima_predictions.png')
plt.show()

# Calcular KPI
y_true = df['price'][last:]
y_pred = best_model.fittedvalues[last:]

r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = root_mean_squared_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

# Crear DataFrame con los KPI
kpi_df = pd.DataFrame({
    'R2': [r2],
    'MAE': [mae],
    'MSE': [mse],
    'RMSE': [rmse],
    'MAPE': [f"{mape:.2f}%"]
})

kpi_df.to_excel('arima_metrics.xlsx', index=False)

print("KPI del modelo ARIMA:")
print(kpi_df)