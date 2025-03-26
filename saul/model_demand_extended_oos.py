import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
import matplotlib.pyplot as plt

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

# 1) Cargar datos
df = pd.read_csv('data.csv', sep=';', parse_dates=['datetime'])
df.sort_values(by='datetime', inplace=True)
df.dropna(subset=['demand'], inplace=True)

# 2) Reservar parte de datos para hold-out (out of sample)
holdout_hours = 24  # Horas out of sample
cutoff = df['datetime'].max() - pd.Timedelta(hours=holdout_hours)
train_df = df[df['datetime'] <= cutoff].copy()
holdout_df = df[df['datetime'] > cutoff].copy()

# 3) Definir características (X) y target (y) para el conjunto de entrenamiento
feature_cols = [c for c in train_df.columns if c not in ['datetime','demand']]
X = train_df[feature_cols]
y = train_df['demand']

# 4) TimeSeriesSplit para validación interna en train_df
tscv = TimeSeriesSplit(n_splits=5)
xgb_scores = []
dt_scores = []
rf_scores = []  # Para almacenar las métricas de RandomForest

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # --------------------- XGBoost ---------------------
    xgb_model = XGBRegressor(random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Métricas training
    xgb_train_preds = xgb_model.predict(X_train)
    xgb_train_r2 = r2_score(y_train, xgb_train_preds)
    xgb_train_mae = mean_absolute_error(y_train, xgb_train_preds)
    xgb_train_mse = mean_squared_error(y_train, xgb_train_preds)
    xgb_train_rmse = root_mean_squared_error(y_train, xgb_train_preds)
    xgb_train_mape = mean_absolute_percentage_error(y_train, xgb_train_preds)

    # Métricas testing
    xgb_test_preds = xgb_model.predict(X_test)
    xgb_test_r2 = r2_score(y_test, xgb_test_preds)
    xgb_test_mae = mean_absolute_error(y_test, xgb_test_preds)
    xgb_test_mse = mean_squared_error(y_test, xgb_test_preds)
    xgb_test_rmse = root_mean_squared_error(y_test, xgb_test_preds)
    xgb_test_mape = mean_absolute_percentage_error(y_test, xgb_test_preds)

    xgb_scores.append((
        xgb_train_r2, xgb_train_mae, xgb_train_mse, xgb_train_rmse, xgb_train_mape,
        xgb_test_r2, xgb_test_mae, xgb_test_mse, xgb_test_rmse, xgb_test_mape
    ))

    # --------------------- Decision Tree ---------------------
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)

    # Métricas training
    dt_train_preds = dt_model.predict(X_train)
    dt_train_r2 = r2_score(y_train, dt_train_preds)
    dt_train_mae = mean_absolute_error(y_train, dt_train_preds)
    dt_train_mse = mean_squared_error(y_train, dt_train_preds)
    dt_train_rmse = root_mean_squared_error(y_train, dt_train_preds)
    dt_train_mape = mean_absolute_percentage_error(y_train, dt_train_preds)

    # Métricas testing
    dt_test_preds = dt_model.predict(X_test)
    dt_test_r2 = r2_score(y_test, dt_test_preds)
    dt_test_mae = mean_absolute_error(y_test, dt_test_preds)
    dt_test_mse = mean_squared_error(y_test, dt_test_preds)
    dt_test_rmse = root_mean_squared_error(y_test, dt_test_preds)
    dt_test_mape = mean_absolute_percentage_error(y_test, dt_test_preds)

    dt_scores.append((
        dt_train_r2, dt_train_mae, dt_train_mse, dt_train_rmse, dt_train_mape,
        dt_test_r2, dt_test_mae, dt_test_mse, dt_test_rmse, dt_test_mape
    ))

    # --------------------- Random Forest (10 árboles) ---------------------
    rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
    rf_model.fit(X_train, y_train)

    # Métricas training
    rf_train_preds = rf_model.predict(X_train)
    rf_train_r2 = r2_score(y_train, rf_train_preds)
    rf_train_mae = mean_absolute_error(y_train, rf_train_preds)
    rf_train_mse = mean_squared_error(y_train, rf_train_preds)
    rf_train_rmse = root_mean_squared_error(y_train, rf_train_preds)
    rf_train_mape = mean_absolute_percentage_error(y_train, rf_train_preds)

    # Métricas testing
    rf_test_preds = rf_model.predict(X_test)
    rf_test_r2 = r2_score(y_test, rf_test_preds)
    rf_test_mae = mean_absolute_error(y_test, rf_test_preds)
    rf_test_mse = mean_squared_error(y_test, rf_test_preds)
    rf_test_rmse = root_mean_squared_error(y_test, rf_test_preds)
    rf_test_mape = mean_absolute_percentage_error(y_test, rf_test_preds)

    rf_scores.append((
        rf_train_r2, rf_train_mae, rf_train_mse, rf_train_rmse, rf_train_mape,
        rf_test_r2, rf_test_mae, rf_test_mse, rf_test_rmse, rf_test_mape
    ))

def resumen_conjunto(scores):
    arr = np.array(scores)
    df_res = pd.DataFrame(
        arr,
        columns=['R2_train','MAE_train','MSE_train', 'RMSE_train', 'MAPE_train(%)',
                 'R2_test','MAE_test','MSE_test', 'RMSE_test', 'MAPE_test(%)']
    )
    df_res.loc['Promedio'] = df_res.mean()
    df_res = df_res.round({
        'R2_train': 4, 'MAE_train': 4, 'MSE_train': 4, 'RMSE_train': 4, 'MAPE_train(%)': 2,
        'R2_test': 4, 'MAE_test': 4, 'MSE_test': 4, 'RMSE_test': 4, 'MAPE_test(%)': 2
    })
    return df_res

print("Resultados XGBoost (validación interna):")
print(resumen_conjunto(xgb_scores))

print("\nResultados Decision Tree (validación interna):")
print(resumen_conjunto(dt_scores))

print("\nResultados Random Forest (validación interna):")
print(resumen_conjunto(rf_scores))

with pd.ExcelWriter('cross_validation_metrics_demanda.xlsx') as writer:
    resumen_conjunto(xgb_scores).to_excel(writer, sheet_name='XGBoost')
    resumen_conjunto(dt_scores).to_excel(writer, sheet_name='Decision Tree')
    resumen_conjunto(rf_scores).to_excel(writer, sheet_name='Random Forest')

# 5) Predicción out of sample en holdout_df
X_holdout = holdout_df[feature_cols]
y_holdout = holdout_df['demand']

xgb_preds_holdout = xgb_model.predict(X_holdout)
dt_preds_holdout = dt_model.predict(X_holdout)
rf_preds_holdout = rf_model.predict(X_holdout)

# 6) Métricas out of sample (Holdout)
xgb_metrics = {
    'Modelo': 'XGBoost',
    'R2': r2_score(y_holdout, xgb_preds_holdout),
    'MAE': mean_absolute_error(y_holdout, xgb_preds_holdout),
    'MSE': mean_squared_error(y_holdout, xgb_preds_holdout),
    'RMSE': root_mean_squared_error(y_holdout, xgb_preds_holdout),
    'MAPE(%)': mean_absolute_percentage_error(y_holdout, xgb_preds_holdout)
}

dt_metrics = {
    'Modelo': 'Decision Tree',
    'R2': r2_score(y_holdout, dt_preds_holdout),
    'MAE': mean_absolute_error(y_holdout, dt_preds_holdout),
    'MSE': mean_squared_error(y_holdout, dt_preds_holdout),
    'RMSE': root_mean_squared_error(y_holdout, dt_preds_holdout),
    'MAPE(%)': mean_absolute_percentage_error(y_holdout, dt_preds_holdout)
}

rf_metrics = {
    'Modelo': 'Random Forest',
    'R2': r2_score(y_holdout, rf_preds_holdout),
    'MAE': mean_absolute_error(y_holdout, rf_preds_holdout),
    'MSE': mean_squared_error(y_holdout, rf_preds_holdout),
    'RMSE': root_mean_squared_error(y_holdout, rf_preds_holdout),
    'MAPE(%)': mean_absolute_percentage_error(y_holdout, rf_preds_holdout)
}

# Crear un DataFrame con las métricas holdout y mostrarlo
metrics_df = pd.DataFrame([xgb_metrics, dt_metrics, rf_metrics])
metrics_df = metrics_df.round({'R2': 4, 'MAE': 4, 'MSE': 4, 'MAPE(%)': 2})

metrics_df.to_excel('holdout_metrics_demanda.xlsx', index=False)

print("\nMétricas Holdout (Out of Sample):")
print(metrics_df)

# 8) Graficar predicciones out of sample
plt.figure(figsize=(10,6))
plt.plot(df['datetime'][-holdout_hours-4:], df['gen_total'][-holdout_hours-4:], label='Real', color='blue')
plt.plot(holdout_df['datetime'], xgb_preds_holdout, label='Prediction XGB', color='red')
plt.plot(holdout_df['datetime'], dt_preds_holdout, label='Prediction DT', color='green')
plt.plot(holdout_df['datetime'], rf_preds_holdout, label='Prediction RF', color='purple')
plt.title(f"Demand - Últimas {holdout_hours} horas (Out of Sample)")
plt.legend()
plt.tight_layout()
plt.savefig('demanda_oos_predictions.png')
plt.show()