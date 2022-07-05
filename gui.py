import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
from NEW.StockPricePrediction import nextdata
import flask
import pandas as pd
import time
import os

server = flask.Flask('app')
server.secret_key = os.environ.get('secret_key', 'secret')

df = pd.read_csv('/Users/abirbadami/Documents/Jupyter/pstock.csv')

app = dash.Dash('app', server=server)

app.scripts.config.serve_locally = False
dcc._js_dist[0]['external_url'] = 'https://cdn.plot.ly/plotly-basic-latest.min.js'

app.layout = html.Div([
    html.H1('Stock Tickers'),
    dcc.Dropdown(
        id='my-dropdown',
        options=[
            {'label': 'Actual Price', 'value': 'REAL'},
            {'label': 'Predicted Price', 'value': 'PRED'},
        ],
        value='REAL'
    ),
    dcc.Graph(id='my-graph'),
    html.Label('Next Day Value'),
    dcc.Textarea(value=nextdata)
], className="container")

@app.callback(Output('my-graph', 'figure'),
              [Input('my-dropdown', 'value')])

def update_graph(selected_dropdown_value):
    if selected_dropdown_value=='REAL':
        return {
            'data': [{
                'x': df.Date,
                'y': df.Predicted,
                'line': {
                    'width': 3,
                    'shape': 'spline'
                }
            }],
            'layout': {
                'margin': {
                    'l': 30,
                    'r': 20,
                    'b': 30,
                    't': 20
                }
            }
        }
    else:
        return {
            'data': [{
                'x': df.Date,
                'y': df.Close,
                'line': {
                    'width': 3,
                    'shape': 'spline',
                    'color':'red'
                }
            }],
            'layout': {
                'margin': {
                    'l': 30,
                    'r': 20,
                    'b': 30,
                    't': 20
                }
            }
        }


if __name__ == '__main__':
    app.run_server()
