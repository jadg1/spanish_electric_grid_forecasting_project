import os
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# Directories
WORK_DIR = "C:/mquea_big_data_/BigDataProject"
RESULTS_DIR = os.path.join(WORK_DIR, "results")

# Dash app initialization
app = dash.Dash(__name__)

# Define a function to load Excel files clearly
def load_forecast_data(filename):
    file_path = os.path.join(RESULTS_DIR, filename)
    df = pd.read_excel(file_path)
    return df

# Load DataFrames from Excel
forecast_data = {
    'gen_total': load_forecast_data('forecast_gen_total.xlsx'),
    'demand': load_forecast_data('forecast_demand.xlsx'),
    'price': load_forecast_data('forecast_price.xlsx')
}

# App Layout
app.layout = html.Div([
    html.H1("Energy Forecast Dashboard", style={'textAlign': 'center'}),
    dcc.Tabs([
        dcc.Tab(label='Generation Total', children=[dcc.Graph(id='gen-total-graph')]),
        dcc.Tab(label='Demand', children=[dcc.Graph(id='demand-graph')]),
        dcc.Tab(label='Price', children=[dcc.Graph(id='price-graph')]),
    ])
])

# Callbacks to generate graphs
@app.callback(
    [Output('gen-total-graph', 'figure'),
     Output('demand-graph', 'figure'),
     Output('price-graph', 'figure')],
    [Input('gen-total-graph', 'id')]
)
def update_graphs(_):
    fig_gen = px.line(forecast_data['gen_total'],
                      title='Generation Total Forecast',
                      labels={'index': 'Hour', 'value': 'Generation (MW)', 'variable': 'Model'})

    fig_demand = px.line(forecast_data['demand'],
                         title='Demand Forecast',
                         labels={'index': 'Hour', 'value': 'Demand (MW)', 'variable': 'Model'})

    fig_price = px.line(forecast_data['price'],
                        title='Price Forecast',
                        labels={'index': 'Hour', 'value': 'Price (€)', 'variable': 'Model'})

    return fig_gen, fig_demand, fig_price

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)