
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import sqlite3
import pandas as pd
from dash.dependencies import Input, Output
from jupyter_dash import JupyterDash
import os

# File paths for SQLite database and results
DB_PATH = '/content/drive/My Drive/BigDataProject/forecast_data.db'
PLOT_DIR = '/content/drive/My Drive/BigDataProject/results/plots'

# Ensure plot directory exists
os.makedirs(PLOT_DIR, exist_ok=True)

# Create the Dash app
app = JupyterDash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Function to retrieve data from SQLite database
def get_forecast_data(target_column, model_name):
    """Retrieve forecast data from the SQLite database for a given target column and model."""
    
    # Connect to SQLite database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Corrected SQL query with no comments inside
    query = """
    SELECT time_step, forecast_value 
    FROM forecast 
    WHERE target_column = ? 
    AND model_name = ? 
    ORDER BY time_step
    """  

    # Execute query and fetch data
    cursor.execute(query, (target_column, model_name))
    data = cursor.fetchall()
    conn.close()
    
    # Separate time_steps and forecast_values
    time_steps = [row[0] for row in data]  
    forecast_values = [row[1] for row in data]  
    
    return time_steps, forecast_values

# Function to plot forecasts
def plot_forecasts(y_true, ml_predictions, target_column="target"):
    """Generate Plotly visualization for actual vs. predicted values."""
    fig = go.Figure()

    # Plot actual values (y_true)
    fig.add_trace(go.Scatter(
        x=y_true.index, y=y_true, mode="lines", 
        name=f"Actual {target_column.upper()}", 
        line=dict(color="black", width=2)
    ))

    # Plot predictions for each model
    for model_name, preds in ml_predictions.items():
        forecast_time_steps = list(range(len(y_true), len(y_true) + len(preds)))
        
        fig.add_trace(go.Scatter(
            x=forecast_time_steps, y=preds, mode="lines", 
            name=f"{model_name} (ML)", 
            line=dict(width=1.5)
        ))

    fig.update_layout(
        title=f"Forecast Comparison for {target_column.upper()}",
        xaxis_title="Time",
        yaxis_title="Value",
        legend_title="Model Used"
    )

    return fig

# Create the dashboard layout
app.layout = html.Div([
    dbc.Tabs([
        # Demand Tab
        dbc.Tab(label="Demand", children=[
            html.H3("Demand Forecasting"),
            dcc.Graph(id='demand-graph')
        ]),
        
        # Generation Tab
        dbc.Tab(label="Generation", children=[
            html.H3("Generation Forecasting"),
            dcc.Graph(id='gen-graph')
        ]),
        
        # Price Tab
        dbc.Tab(label="Price", children=[
            html.H3("Price Forecasting"),
            dcc.Graph(id='price-graph')
        ]),
        
        # Demand vs Generation Difference Tab
        dbc.Tab(label="Demand & Generation Difference", children=[
            html.H3("Demand vs Generation Forecast Difference"),
            dcc.Graph(id='diff-graph')
        ])
    ])
])

# Callback to update graphs
@app.callback(
    [Output('demand-graph', 'figure'),
     Output('gen-graph', 'figure'),
     Output('price-graph', 'figure'),
     Output('diff-graph', 'figure')],
    [Input('demand-graph', 'id')]  # Input to trigger update
)
def update_graphs(input_data):
    """Fetch forecast data and update plots."""
    
    TARGET_COLUMNS = ["demand", "gen_total", "price"]
    
    y_holdout_dict = {}
    ml_predictions_dict = {}
    diff_dict = {}
    
    for target_column in TARGET_COLUMNS:
        print(f"🔄 Loading data for {target_column}")
        
        # Load actual data from the database
        time_steps, actual_data = get_forecast_data(target_column, "XGBoost")  
        y_holdout_dict[target_column] = pd.Series(actual_data, index=time_steps)
        
        # Load forecast results for different models
        forecast_data = {model_name: get_forecast_data(target_column, model_name)[1] 
                         for model_name in ["XGBoost", "DecisionTree", "RandomForest"]}
        ml_predictions_dict[target_column] = forecast_data

        # Calculate difference between demand and generation (for "demand" and "gen_total")
        if target_column == "demand" or target_column == "gen_total":
            diff = y_holdout_dict[target_column] - forecast_data["XGBoost"]
            diff_dict[target_column] = diff

    # Plot for Demand
    demand_fig = plot_forecasts(y_holdout_dict.get("demand", pd.Series([])), 
                                ml_predictions_dict.get("demand", {}), target_column="demand")
    
    # Plot for Generation
    gen_fig = plot_forecasts(y_holdout_dict.get("gen_total", pd.Series([])), 
                             ml_predictions_dict.get("gen_total", {}), target_column="gen_total")
    
    # Plot for Price
    price_fig = plot_forecasts(y_holdout_dict.get("price", pd.Series([])), 
                               ml_predictions_dict.get("price", {}), target_column="price")
    
    # Plot for Demand vs Generation Difference
    diff_fig = go.Figure()
    if "demand" in diff_dict:
        diff_fig.add_trace(go.Scatter(
            x=y_holdout_dict.get("demand", pd.Series([])).index, 
            y=diff_dict.get("demand", pd.Series([])), mode="lines", 
            name="Demand - Generation Difference",
            line=dict(color="blue")
        ))

    # Return the figures
    return demand_fig, gen_fig, price_fig, diff_fig

# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)  # Run the app in Colab with no reloader

