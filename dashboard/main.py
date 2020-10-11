import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import pyodbc
import scipy.integrate as integrate
import scipy.special as special
from dash.dependencies import Input, Output
import torch
# all_teams_df = pd.read_csv('srcdata/shot_dist_compiled_data_2019_20.csv')
from flask import Flask, jsonify
from flask_restful import Resource, Api

from setup_data import create_data, retreive_db_data
from utils import load_tree, load_model, load_ridge

tree = load_tree("../regression_tree_model")
ridge = load_ridge("../ridge_model")
nn = load_model("../model")

conn = pyodbc.connect(DRIVER='{ODBC Driver 17 for SQL Server}',
                      SERVER='devohackaton.database.windows.net',
                      DATABASE='privilege',
                      UID='aadmin',
                      PWD='9rTuUjFcssBUGws2')

cursor = conn.cursor()

details = pd.read_sql_query('SELECT * FROM [privilege].[dbo].[userdetailtrains]', conn)
del details["CreatedByUserId"]
# del data["UserId"]
del details["CreditScore"]
del details["CreationDate"]
del details["Id"]

# noisy data
del details["Income"]
del details["JobClass"]
del details["EmploymentType"]
del details["SocialStability"]
del details["SocialExposure"]
del details["SocialQuality"]
print(list(details.columns.values))

columns = list(details.columns.values)

columns.remove("UserId")
columns.remove("LastName")
columns.remove("FirstName")

types = ["number"] * len(columns)

server = Flask('my_app')
# server = Flask('my_app')
app = dash.Dash(server=server)

# app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(
    [
        html.Div([dcc.Dropdown(id='group-select',
                               options=[{'label': firstname + " " + lastname, 'value': userid} for
                                        firstname, lastname, userid in
                                        zip(details["LastName"], details["FirstName"], details["UserId"])],
                               value='', style={'width': '300px'})]),
        html.Div([dcc.Dropdown(id='model-select',
                               options=[{'label': model, 'value': model} for
                                        model in ["RegressionTree","Ridge", "NeuralNetwork"]],
                               value='RegressionTree', style={'width': '300px'})]),
        dcc.Graph('shot-dist-graph', config={'displayModeBar': False})
    ]
    # +
    # [
    #     dcc.Input(
    #         id="input_{}".format(col),
    #         type=coltype,
    #         placeholder=col,
    #     )
    #     for col, coltype in zip(columns, types)
    # ]
)


@app.callback(
    Output('shot-dist-graph', 'figure'),
    [Input('group-select', 'value'), Input("model-select", "value")]
)
def update_graph(user_id, model_type):
    import plotly.express as px
    if user_id == "":
        X_graph = [4, 5, 6]
        Y = [4, 5, 6]
        credit_score = 0.
    else:
        X, Y, X_graph = compute(user_id, model_type)
        credit_score = compute_creditscore(user_id)
    data = []
    for x, y in zip(X_graph, Y):
        data.append([x, y])

    title="Likelihood or reimbursement (Credit score: {:.2f})".format(credit_score[0])
    # title="Likelihood or reimbursement (Credit score: )" #.format(credit_score)
    df = pd.DataFrame(data, columns=["Amount lend", "Probability"])
    fig = px.line(df, x="Amount lend", y="Probability",
                  title=title, log_x=True)
    fig.update_layout(
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )
    fig.update_yaxes(range=[0, 1])
    return fig


def compute(user_id, model_type, amounts=[10, 100, 500, 1000, 5000, 7500, 10000, 50000, 100000]):
    print("user_id: {}".format(user_id))
    X = details[details["UserId"] == user_id]
    # X = details[details["UserId"] == "DF5A07B6-2880-4411-84F1-5F62153B0882"]
    del X["UserId"]
    del X["LastName"]
    del X["FirstName"]
    X = X.replace("M", 0).replace("F", 1).replace(True, 1).replace(False, 0)
    print(X)
    userinfo = list(X.values[0])
    X = []

    X_graph = []
    for amount in amounts:
        X_graph.append(amount)
        X.append(userinfo + [amount, ])
    if model_type == "RegressionTree":
        Y = tree.predict(X)
    elif model_type == "NeuralNetwork":
        Y = nn(torch.Tensor(X)).squeeze().tolist()
    elif model_type == "Ridge":
        Y = ridge.predict(X)
    else:
        pass
    return X, Y, X_graph


def compute_creditscore(username):
    amounts = [10, 100, 500, 1000, 5000, 7500, 10000, 50000, 100000]
    X, Y, X_graph = compute(username, "RegressionTree", amounts)
    from sklearn import linear_model
    reg = linear_model.Ridge(alpha=.5)
    reg.fit(np.array(X_graph).reshape(-1, 1), Y)
    result = integrate.quad(lambda x: reg.predict([[x]]), 0, 100000)
    return result


@server.route('/api/<username>')
def meteo(username):
    amounts = [10, 100, 500, 1000, 5000, 7500, 10000, 50000, 100000]
    X, Y, X_graph = compute(username, "RegressionTree", amounts)
    dictionnaire = {
        'probabilities': Y.tolist(),
        'amounts': amounts,
    }
    return jsonify(dictionnaire)


import numpy as np
@server.route('/creditscore/<username>')
def credit_score(username):
    dictionnaire = {
        'credit_score': compute_creditscore(username),
    }
    return jsonify(dictionnaire)


if __name__ == '__main__':
    app.run_server(debug=False, port=8080, host="0.0.0.0")
