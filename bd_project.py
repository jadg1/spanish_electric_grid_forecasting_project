# ---- Imports ----
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import joblib

# ---- Global Paths and Directory Setup ----
WORK_DIR = "C:/mquea_big_data_/BigDataProject"
MODEL_DIR = os.path.join(WORK_DIR, "saved_models")
PLOT_DIR = os.path.join(WORK_DIR, "results", "plots")
TARGET_COLUMNS = ["demand", "gen_total", "price"]
RESULTS_DIR = os.path.join(WORK_DIR, "results")
DATA_FILE = os.path.join(WORK_DIR, "data.csv")
TARGET_COLUMNS = ["demand", "gen_total", "price"]
RESULTS_DIR = os.path.join(WORK_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(f"{WORK_DIR}/results", exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.chdir(WORK_DIR)

files = os.listdir(WORK_DIR)
print("📂 Files in project directory:", files)

# Load data if file exists
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE, sep=';', parse_dates=['datetime'])
    print("✅ Data loaded successfully!")
    print(df.head())  # Show first rows
else:
    print("❌ Data file not found! Upload 'data.csv'.")

# ---- Data Loading Functions ----
def load_data(file_path, target_column):
    """Load dataset and prepare datetime index."""
    df = pd.read_csv(file_path, sep=';', parse_dates=['datetime'])
    df.sort_values(by='datetime', inplace=True)
    df.set_index('datetime', inplace=True)
    df.dropna(subset=[target_column], inplace=True)
    return df

# ---- ML Model Training and Saving Functions ----
def ml_train_test_split(df, target_column, holdout_hours=24):
    """Splits data into training and holdout set for ML models."""
    cutoff = df.index.max() - pd.Timedelta(hours=holdout_hours)
    train_df = df[df.index <= cutoff]
    holdout_df = df[df.index > cutoff]

    feature_cols = [col for col in df.columns if col != target_column]
    X_train, y_train = train_df[feature_cols], train_df[target_column]
    X_holdout, y_holdout = holdout_df[feature_cols], holdout_df[target_column]

    return X_train, y_train, X_holdout, y_holdout

def train_ml_models(X_train, y_train, target_column):
    """Train ML models and save them to Google Drive if not already saved."""
    model_paths = {
        'XGBoost': os.path.join(MODEL_DIR, f"xgboost_{target_column}.joblib"),
        'DecisionTree': os.path.join(MODEL_DIR, f"decision_tree_{target_column}.joblib"),
        'RandomForest': os.path.join(MODEL_DIR, f"random_forest_{target_column}.joblib")
    }

    models = {
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
        'DecisionTree': DecisionTreeRegressor(random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=20, random_state=42)
    }

    trained_models = {}
    for name, model in models.items():
        if os.path.exists(model_paths[name]):  # Check if model exists
            print(f"🔄 Loading saved {name} model for {target_column}...")
            trained_models[name] = joblib.load(model_paths[name])
            print(f"✅ Successfully loaded {name} model for {target_column}.")
        else:
            print(f"⚡ Training {name} for {target_column}...")
            model.fit(X_train, y_train)
            joblib.dump(model, model_paths[name])  # Save model
            trained_models[name] = model
            print(f"✅ {name} model saved: {model_paths[name]}")

    return trained_models

def cross_validate(X, y, models, n_splits=5):
    """Perform time series cross-validation and return results."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = {name: [] for name in models}

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results[name].append({
                "R2": r2_score(y_test, y_pred),
                "MAE": mean_absolute_error(y_test, y_pred),
                "MSE": mean_squared_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
            })

    return results

def arima_train_test_split(df, target_column, holdout_hours=24):
    """Splits data into train and holdout for ARIMA (Univariate)."""
    cutoff = df.index.max() - pd.Timedelta(hours=holdout_hours)
    train_series = df.loc[df.index <= cutoff, target_column]
    holdout_series = df.loc[df.index > cutoff, target_column]

    return train_series, holdout_series

def train_arima(y_train, target_column, p_range=(0,3), d_range=(0,2), q_range=(0,3)):
    """Train ARIMA model and save it to Google Drive if not already saved."""
    model_path = os.path.join(MODEL_DIR, f"arima_{target_column}.joblib")

    # 🔄 Check if an ARIMA model is already saved, and load it
    if os.path.exists(model_path):
        print(f"🔄 Loading saved ARIMA model for {target_column}...")
        return joblib.load(model_path)

    # 🚀 Perform ARIMA hyperparameter tuning if no saved model is found
    best_aic, best_order, best_model = float('inf'), None, None
    for p in range(*p_range):
        for d in range(*d_range):
            for q in range(*q_range):
                try:
                    model = ARIMA(y_train, order=(p,d,q)).fit()
                    if model.aic < best_aic:
                        best_aic, best_order, best_model = model.aic, (p,d,q), model
                except:
                    continue

    print(f"✅ Best ARIMA Order for {target_column}: {best_order} with AIC: {best_aic}")

    # 💾 Save the trained ARIMA model
    joblib.dump(best_model, model_path)
    print(f"✅ ARIMA model saved: {model_path}")

    return best_model

def forecast_models(models, X_holdout):
    """Generate ML-based forecasts."""
    return {name: model.predict(X_holdout) for name, model in models.items()}

def forecast_arima(arima_model, steps=12):
    """Generate ARIMA forecasts."""
    return arima_model.forecast(steps=steps)

# ---- Anomaly Detection Function ----
def detect_and_plot_anomalies(filepath, date_col='Date', value_col='price', contamination=0.05):
    df = pd.read_excel(filepath)
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df.dropna(subset=[value_col], inplace=True)

    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    df['anomaly'] = iso_forest.fit_predict(df[[value_col]])

    anomalies = df[df['anomaly'] == -1]
    normal = df[df['anomaly'] == 1]

    plt.figure(figsize=(12, 6))
    plt.plot(normal[date_col], normal[value_col], color="blue", linestyle='-', label="Normal Data")
    plt.scatter(anomalies[date_col], anomalies[value_col], color="red", label="Anomalies")
    plt.title("Anomaly Detection")
    plt.xlabel("Date")
    plt.ylabel(value_col)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return anomalies

# ---- Visualization Functions ----
def plot_generation_demand_diff(df):
    df['generation_demand_diff'] = df['gen_total'] - df['demand']
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['generation_demand_diff'], label='Generation - Demand', color='navy')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('Datetime')
    plt.ylabel('Generation-Demand (MW)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_diff_vs_price(df):
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(df.index, df['generation_demand_diff'], 'b-', label='Gen-Demand Diff')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Generation-Demand (MW)', color='b')

    ax2 = ax1.twinx()
    ax2.plot(df.index, df['price'], 'r-', label='Price (€)')
    ax2.set_ylabel('Price (€)')

    fig.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_forecasts(y_true, ml_predictions, arima_predictions=None, target_column="target"):
    """Plot actual vs. predicted values for all ML models + ARIMA (if applicable)."""
    fig = go.Figure()

    # Plot Actual Values
    fig.add_trace(go.Scatter(
        x=y_true.index, y=y_true, mode="lines",
        name=f"Actual {target_column.upper()}", line=dict(color="black", width=2)
    ))

    # Plot ML Predictions (Solid Lines)
    for model_name, preds in ml_predictions.items():
        fig.add_trace(go.Scatter(
            x=y_true.index, y=preds, mode="lines",
            name=f"{model_name} (ML)", line=dict(width=1.5)
        ))

    # Plot ARIMA Predictions (Dashed Line, Only for Price)
    if arima_predictions is not None:
        fig.add_trace(go.Scatter(
            x=y_true.index, y=arima_predictions, mode="lines",
            name="ARIMA (Time Series)", line=dict(dash='dot', width=2, color="red")
        ))

    fig.update_layout(
        title=f"Forecast Comparison for {target_column.upper()}",
        xaxis_title="Time",
        yaxis_title="Value",
        legend_title="Model Used"
    )

    # Display the plot interactively
    fig.show()

    return fig  # Return the figure for any future use

# ---- Forecast saving and loading ----
def save_results(results, file_path):
    """Save model performance metrics to an Excel file."""
    with pd.ExcelWriter(file_path) as writer:
        for model, scores in results.items():
            pd.DataFrame(scores).to_excel(writer, sheet_name=model)
    print(f"✅ Model performance saved at: {file_path}")

def save_forecasts(ml_predictions, arima_predictions, file_path):
    """Save ML & ARIMA forecast results to an Excel file."""
    with pd.ExcelWriter(file_path) as writer:
        # Save ML predictions
        pd.DataFrame(ml_predictions).to_excel(writer, sheet_name="ML_Forecasts")

        # Save ARIMA predictions (only for 'price')
        if arima_predictions is not None:
            pd.DataFrame({'ARIMA': arima_predictions}).to_excel(writer, sheet_name="ARIMA_Forecasts")

    print(f"✅ Forecast results saved at: {file_path}")

def load_forecasts(file_path):
    """Load saved forecasts from an Excel file."""
    if os.path.exists(file_path):
        print(f"🔄 Loading forecasts from {file_path}...")
        return pd.read_excel(file_path, sheet_name=None)
    else:
        print(f"❌ Forecast file {file_path} not found!")
        return None
    
# ---- Run all models ----

def run_all_models(file_path, target_columns):
    """Run ML and ARIMA models for each target variable, save results, and return predictions for visualization."""

    all_y_holdout = {}
    all_ml_predictions = {}
    all_y_holdout_arima = {}
    all_arima_predictions = {}
    all_cv_results = {}

    for target_column in target_columns:
        print(f"\n🚀 Processing target: {target_column.upper()}")

        df = load_data(file_path, target_column)

        # ML Models
        X_train, y_train, X_holdout, y_holdout = ml_train_test_split(df, target_column)
        ml_models = train_ml_models(X_train, y_train, target_column)
        cv_results = cross_validate(X_train, y_train, ml_models)

        # ✅ Save cross-validation results
        cv_results_path = os.path.join(RESULTS_DIR, f"cv_results_{target_column}.xlsx")
        save_results(cv_results, cv_results_path)

        # ✅ Generate ML Predictions
        ml_predictions = forecast_models(ml_models, X_holdout)

        # Store ML Predictions & Actual Values
        all_y_holdout[target_column] = y_holdout
        all_ml_predictions[target_column] = ml_predictions
        all_cv_results[target_column] = cv_results

        # ARIMA Model (Only for `price`)
        if target_column == "price":
            y_train_arima, y_holdout_arima = arima_train_test_split(df, target_column)
            arima_model = train_arima(y_train_arima, target_column)
            arima_predictions = forecast_arima(arima_model, steps=len(y_holdout_arima))

            # Store ARIMA results
            all_y_holdout_arima[target_column] = y_holdout_arima
            all_arima_predictions[target_column] = arima_predictions
        else:
            arima_predictions = None  # Assign None for non-price targets

        # ✅ Save ML and ARIMA forecasts
        forecast_path = os.path.join(RESULTS_DIR, f"forecast_{target_column}.xlsx")
        save_forecasts(ml_predictions, arima_predictions, forecast_path)

        print(f"✅ Completed processing for {target_column}. Results saved in {RESULTS_DIR}\n")

    return all_y_holdout, all_ml_predictions, all_y_holdout_arima, all_arima_predictions, all_cv_results

# ---- Plots ----

def plot_raw_data(df, target_column):
    """Plot raw time series data."""
    fig = px.line(df, x=df.index, y=target_column, title=f"{target_column.upper()} Over Time")
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title=f"{target_column.upper()} Value",
        template="plotly_dark"
    )
    fig.show()

def plot_forecasts(y_true, ml_predictions, arima_predictions=None, target_column="target"):
    """Plot actual vs. predicted values for all ML models + ARIMA (if applicable)."""
    fig = go.Figure()

    # Plot Actual Values
    fig.add_trace(go.Scatter(
        x=y_true.index, y=y_true, mode="lines",
        name=f"Actual {target_column.upper()}", line=dict(color="black", width=2)
    ))

    # Plot ML Predictions (Solid Lines)
    for model_name, preds in ml_predictions.items():
        fig.add_trace(go.Scatter(
            x=y_true.index, y=preds, mode="lines",
            name=f"{model_name} (ML)", line=dict(width=1.5)
        ))

    # Plot ARIMA Predictions (Dashed Line, Only for Price)
    if arima_predictions is not None:
        fig.add_trace(go.Scatter(
            x=y_true.index, y=arima_predictions, mode="lines",
            name="ARIMA (Time Series)", line=dict(dash='dot', width=2, color="red")
        ))

    fig.update_layout(
        title=f"Forecast Comparison for {target_column.upper()}",
        xaxis_title="Time",
        yaxis_title="Value",
        legend_title="Model Used"
    )

    # Display the plot interactively
    fig.show()

    return fig  # Return the figure for any future use

def plot_model_performance(results, metric="R2"):
    """Visualize model performance across cross-validation folds."""
    # Create a DataFrame with the cross-validation results
    df_results = pd.DataFrame({model: [run[metric] for run in scores] for model, scores in results.items()})

    # Create a boxplot to show model performance
    fig = px.box(df_results, title=f"{metric} Performance Across Models",
                 labels={'value': metric, 'variable': 'Model'},
                 template="plotly_dark")

    # Display the plot interactively (without saving it)
    fig.show()

    return fig  # Return figure for saving if needed later

def save_plot(fig, filename):
    """Save Plotly figure as an image in Google Drive."""
    file_path = os.path.join(PLOT_DIR, filename)
    fig.write_image(file_path)
    print(f"✅ Plot saved: {file_path}")

def generate_and_display_plots(y_holdout_dict, ml_predictions_dict, cv_results_dict, arima_predictions_dict=None):
    """Generate and display all relevant plots from forecasting results."""

    for target_column in y_holdout_dict.keys():
        print(f"\n📈 Generating plots for: {target_column.upper()}")

        y_holdout = y_holdout_dict.get(target_column)
        ml_predictions = ml_predictions_dict.get(target_column)

        # Plot cross-validation metrics (if available)
        cv_results = cv_results_dict.get(target_column)
        if cv_results:
            plot_model_performance(cv_results, metric="R2")
            plot_model_performance(cv_results, metric="MAE")
            plot_model_performance(cv_results, metric="RMSE")

        # Plot ML forecasts
        if y_holdout is not None and ml_predictions:
            plot_forecasts(y_holdout, ml_predictions, target_column=target_column)

        # Include ARIMA forecasts for 'price' if available
        if target_column == "price" and arima_predictions_dict:
            arima_predictions = arima_predictions_dict.get(target_column)
            if arima_predictions is not None:
                plot_forecasts(y_holdout_dict[target_column], ml_predictions_dict[target_column], arima_predictions, target_column=target_column)


# ---- Anomalies ----

def plot_diff_vs_price(df):
    fig, ax1 = plt.subplots(figsize=(15, 7))

    # Plot generation-demand difference
    ax1.plot(df.index, df['generation_demand_diff'], color='navy', label='Generation - Demand Difference (MW)')
    ax1.axhline(0, color='grey', linestyle='--', linewidth=1)
    ax1.set_xlabel('Datetime', fontsize=14)
    ax1.set_ylabel('Generation-Demand Difference (MW)', fontsize=14, color='navy')
    ax1.tick_params(axis='y', labelcolor='navy')

    # Secondary y-axis for price
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['price'], color='orange', alpha=0.7, label='Electricity Price (€)')
    ax2.set_ylabel('Price (€)', fontsize=14, color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Titles and legends
    plt.title('Graphical Explanation: Generation-Demand Difference and Electricity Price', fontsize=16)
    fig.tight_layout()
    fig.autofmt_xdate()

    # Legends for both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

# ---- Main Function (Good Practice) ----

def main():
    DATA_FILE = os.path.join(WORK_DIR, "data.csv")
    GAS_FILTRADO_FILE = os.path.join(WORK_DIR, "GAS_FILTRADO.xlsx")

    # Check data file existence
    if not os.path.exists(DATA_FILE):
        print("❌ data.csv missing!")
        return

    # Load and preview data
    df = load_data(DATA_FILE, target_column="demand")
    print("✅ Data loaded successfully!")
    print(df.head())

    # Run all models and forecasts
    y_holdout, ml_predictions, y_holdout_arima, arima_predictions, cv_results = run_all_models(DATA_FILE, TARGET_COLUMNS)

    # Generate and display all forecast and model performance plots
    generate_and_display_plots(y_holdout, ml_predictions, cv_results, arima_predictions)

    # Calculate and visualize generation-demand difference
    plot_generation_demand_diff(df)

    # Plot difference versus price
    df['generation_demand_diff'] = df['gen_total'] - df['demand']
    plot_diff_vs_price(df)

    # Anomaly detection
    if os.path.exists(GAS_FILTRADO_FILE):
        anomalies = detect_and_plot_anomalies(GAS_FILTRADO_FILE, date_col='Date', value_col='price', contamination=0.05)
        print("✅ Anomalies detected:")
        print(anomalies.head())
    else:
        print("❌ GAS_FILTRADO.xlsx missing.")

if __name__ == "__main__":
    main()
