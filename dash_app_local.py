import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import os
from dash.dependencies import Input, Output

# Local paths
WORK_DIR = "C:/mquea_big_data_/BigDataProject/results"
FORECAST_FILES = {
    "demand": os.path.join(WORK_DIR, "forecast_demand.xlsx"),
    "gen_total": os.path.join(WORK_DIR, "forecast_gen_total.xlsx"),
    "price": os.path.join(WORK_DIR, "forecast_price.xlsx")
}

# Load data from Excel
def load_forecast_data(target_column):
    file_path = FORECAST_FILES.get(target_column)
    if not file_path or not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return pd.DataFrame()
    df = pd.read_excel(file_path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors='coerce')
    return df

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Function to plot forecasts
def plot_forecasts(df, target_column):
    fig = go.Figure()
    if not df.empty:
        fig.add_trace(go.Scatter(x=df["datetime"], y=df["actual"], mode="lines", name=f"Actual {target_column.upper()}", line=dict(color="black", width=2)))
        for col in df.columns:
            if col.startswith("forecast_"):
                fig.add_trace(go.Scatter(x=df["datetime"], y=df[col], mode="lines", name=col, line=dict(width=1.5)))
    fig.update_layout(title=f"Forecast Comparison for {target_column.upper()}", xaxis_title="Time", yaxis_title="Value", legend_title="Model Used")
    return fig

# Dashboard layout
app.layout = html.Div([
    dbc.Tabs([
        dbc.Tab(label="Demand", children=[html.H3("Demand Forecasting"), dcc.Graph(id='demand-graph')]),
        dbc.Tab(label="Generation", children=[html.H3("Generation Forecasting"), dcc.Graph(id='gen-graph')]),
        dbc.Tab(label="Price", children=[html.H3("Price Forecasting"), dcc.Graph(id='price-graph')]),
        dbc.Tab(label="Demand & Generation Difference", children=[html.H3("Demand vs Generation Forecast Difference"), dcc.Graph(id='diff-graph')])
    ])
])

# Callback to update graphs
@app.callback(
    [Output('demand-graph', 'figure'),
     Output('gen-graph', 'figure'),
     Output('price-graph', 'figure'),
     Output('diff-graph', 'figure')],
    [Input('demand-graph', 'id')]
)
def update_graphs(_):
    demand_df = load_forecast_data("demand")
    gen_df = load_forecast_data("gen_total")
    price_df = load_forecast_data("price")
    diff_df = demand_df.copy()
    if not demand_df.empty and not gen_df.empty:
        diff_df["diff"] = demand_df["actual"] - gen_df["actual"]
    demand_fig = plot_forecasts(demand_df, "demand")
    gen_fig = plot_forecasts(gen_df, "gen_total")
    price_fig = plot_forecasts(price_df, "price")
    diff_fig = go.Figure()
    if not diff_df.empty:
        diff_fig.add_trace(go.Scatter(x=diff_df["datetime"], y=diff_df["diff"], mode="lines", name="Demand - Generation Difference", line=dict(color="blue")))
    return demand_fig, gen_fig, price_fig, diff_fig

# Run the app locally
if __name__ == "__main__":
    app.run_server(debug=True)