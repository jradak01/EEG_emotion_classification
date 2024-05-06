# Run `py app.py` in vscode
# visit http://127.0.0.1:8050/ in web browser

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import csv
import numpy as np
import sys
import os

import reader
import model

#restart app
def restart():
    print("argv was",sys.argv)
    print("sys.executable was", sys.executable)
    print("restart now")

    os.execv(sys.executable, ['python'] + sys.argv)

#create dash app
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])

#used colors
colors = {
    'black':'#000000',
    'lightblack': '#1B1B1B',
    'shadow':'#333333',
    'grey': '#808080',
    'lightgrey': '#ababab',
    'lightblue': '#58a4b0',
    'white': '#f1f1f1',
    'darkgreyblue': '#232734'
}

##app
app.layout = dbc.Container(children=[
    dbc.Row([
        #upload-show image
        dbc.Col([
            dbc.Card(dbc.CardBody([

                dcc.Upload(
                id='upload-image',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px',
                    'color': colors['white']
                },
                multiple=True
                )]), color=colors['lightblack'], style={'marginTop': 10, 'marginBottom': 10}),
                html.Div(id='output-image-upload', style={"textAlign":'center'}),
        ]),
        #live wave graph
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    dcc.Graph(id='live-update-graph', animate=True)
                ),color=colors['lightblack'], 
                style={'marginTop': 10, 'marginBottom': 10}
            )
        ),
        dcc.Interval(id='interval-component', interval=1*1000, n_intervals=0)
    ]),
    dbc.Row([
        #emotion prediction
        dbc.Col(
            dbc.Card(dbc.CardBody([
                html.Div(
                    html.I(className="bi bi-emoji-laughing-fill"),
                    style={"color": colors["shadow"], "fontSize": 69},
                    id='happy'
                ),
                # html.Div(
                #     html.I(className="bi bi-emoji-neutral-fill"),
                #     style={"color": colors["lightblue"], "fontSize": 46},
                #     id='neutral'
                # ),
                html.Div(
                    html.I(className="bi bi-emoji-frown-fill"),
                    style={"color": colors["shadow"], "fontSize": 69},
                    id='sad'
                ),
                dbc.Button(html.I(className="bi bi-play"),
                    id="btn-start-prediction", n_clicks=0, color="light", size="lg",
                    style={"backgroundColor": colors["lightblue"], "color": colors["lightblack"], "border":"none", "marginTop": 16, "marginBottom": 10}
                ),
                dcc.Interval(id='interval-component3', interval=1*1000, n_intervals=0, disabled=True, max_intervals=-1),
        ],), color=colors['lightblack'], style={"textAlign": "center"})
        ),
        #commands for data collection
        dbc.Col([
            dbc.Card(dbc.CardBody([
                dbc.RadioItems(
                    options=[
                        {"label": "Pozitivno", "value": 'pozitivno'},
                        {"label": "Neutralno", "value": 'neutralno', "disabled": True},
                        {"label": "Negativno", "value": 'negativno'},
                    ],
                    value='neutralno',
                    label_style={"color": colors['white'], 'fontSize': 20},
                    input_style={
                        "backgroundColor": colors['black'],
                        "borderColor": colors['black'],
                    },
                    label_checked_style={"color": colors['lightblue']},
                    input_checked_style={
                        "backgroundColor": colors['lightblue'],
                        "borderColor": colors['lightblack'],
                    },
                    style={"marginTop": 40, "marginBottom": 40},
                    id="radioitems-input"
                ),
                dcc.Interval(id='interval-component2', interval=1*1000, n_intervals=0, disabled=True, max_intervals=8),
                dbc.ButtonGroup([
                    dbc.Button(html.I(className="bi bi-play"), 
                        id="btn-play", n_clicks=0, color="light",
                        style={"backgroundColor": colors["lightblue"], "color": colors["lightblack"], "border":"none"}
                    ), 
                    dbc.Button(html.I(className="bi bi-download"), 
                        id="btn-download", n_clicks=0, color="light", 
                        style={"backgroundColor": colors["lightblue"], "color": colors["lightblack"], "border":"none"}
                    ), 
                    dbc.Button(html.I(className="bi bi-arrow-right"), 
                        id="btn-forward", n_clicks=0, color="light", 
                        style={"backgroundColor": colors["lightblue"], "color": colors["lightblack"], "border":"none"}
                    )
                ], size="lg",style={"marginTop": 10, "marginBottom": 10}),
                html.Div(id='output-csv'),
                html.Div(id='download-csv', style={"marginTop": 10, "marginBottom": 10}),
            ]),
            color=colors['lightblack'], style={"textAlign":"center"})
        ]),
        #live attention gauge
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    dcc.Graph(id='live-update-attention', animate=True)
                ), color=colors['lightblack']
            )
        ),
        #live meditation gauge
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    dcc.Graph(id='live-update-meditation', animate=True)
                ), color=colors['lightblack']
            )
        )
    ]),
    dbc.Row([
        dbc.Col([
            #show modal
            dbc.Button(html.I(className='bi bi-eye'),
                id="btn-model", n_clicks=0,color="dark",
                style={"backgroundColor": colors["lightblack"], "color": colors["lightblue"], "border":"none", 'fontSize':20}
            ),
            #connect mindwave
            dbc.Button(html.I(className='bi bi-headset'),
                id="btn-connect", n_clicks=0,color="dark",
                style={"backgroundColor": colors["lightblack"], "color": colors["lightblue"], "border":"none", 'fontSize':20}
            ),
            #disconnect mindwave and restart
            dbc.Button(html.I(className='bi bi-slash-square'),
                id="btn-disconnect", n_clicks=0,color="dark",
                style={"backgroundColor": colors["lightblack"], "color": colors["lightblue"], "border":"none", 'fontSize':20}
            ),
            #refresh page
            html.A(dbc.Button(html.I(className='bi bi-arrow-clockwise'),
                id="btn-refresh-page", n_clicks=0,color="dark",
                style={"backgroundColor": colors["lightblack"], "color": colors["lightblue"], "border":"none", 'fontSize':20}), 
                href='/' 
            ),
            #modal
            dbc.Modal(
                [
                    #confusion matrix
                    dbc.ModalBody(
                        dcc.Graph(id="knn-confusion"), 
                        style={"backgroundColor": colors["lightblack"], "color":colors["white"],"border":'none'}
                    ),
                    #accuracy graph
                    dbc.ModalBody(
                        dcc.Graph(id="knn-accuracy"), 
                        style={"backgroundColor": colors["lightblack"], "color":colors["white"],"border":'none'}
                    ),
                    #slider - dertemine k
                    dbc.ModalBody(
                        dcc.Slider(1, 20, 1, value=3, marks=None,
                        tooltip={"placement": "bottom", "always_visible": True}, id="slider_k"), 
                        style={"backgroundColor": colors["lightblack"], "color":colors["white"],"border":'none'}
                    ),
                    #make new model
                    dbc.ModalBody(
                        dbc.Button("Postavi",id="btn-change-k", n_clicks=0,color="dark",
                        style={"backgroundColor": colors["lightblack"], "color": colors["lightblue"], "border":"none", 'fontSize':20}), style={"backgroundColor": colors["lightblack"], "color":colors["white"],"border":'none', "textAlign":'center'}
                    ),
                    #close modal
                    dbc.ModalFooter(
                        dbc.Button(
                            "Zatvori", id="close", className="ms-auto", n_clicks=0, color="dark",style={"backgroundColor": colors["lightblack"], "color": colors["lightblue"], "border":"none", 'fontSize':20}
                        ), style={"backgroundColor": colors["lightblack"], "color":colors["white"],"border":'none'}
                    ),
                ],id="modal",is_open=False
            ),
            html.Div(id="connect-output"),
            html.Div(id="disconnect-output")
        ], style={"textAlign":"right"})
    ], style={'marginTop':10} ),
], style={'backgroundColor': colors['black']})

#adding photo to view
def parse_contents(contents, filename, date):
    return html.Div([
        html.Img(src=contents, style={'height':'60%', 'width':'60%'}),
    ])

#live updates of live graphs - waves, attention, meditation
@app.callback([Output('live-update-graph', 'figure'), 
                Output('live-update-attention', 'figure'), 
                Output('live-update-meditation', 'figure')],
                Input('interval-component', 'n_intervals'))
def get_live_updates(n):
    df = pd.read_csv('data.csv')
    mode = 'lines'
    fig1 = px.scatter()
    fig1.add_scatter(name='Delta', y=df['delta'], mode=mode)
    fig1.add_scatter(name='Theta', y=df['theta'], mode=mode)
    fig1.add_scatter(name='LowAlpha', y=df['lowalpha'], mode=mode)
    fig1.add_scatter(name='HighAlpha', y=df['highalpha'], mode=mode)
    fig1.add_scatter(name='LowBeta', y=df['lowbeta'], mode=mode)
    fig1.add_scatter(name='HighBeta', y=df['highbeta'], mode=mode)
    fig1.add_scatter(name='LowGamma', y=df['lowgamma'], mode=mode)
    fig1.add_scatter(name='HighGamma', y=df['highgamma'], mode=mode)
    fig1.update_xaxes(range = [0,len(df)-1])
    fig1.update_layout({"margin": {"l": 0, "r": 0, "b": 0, "t": 20}, "autosize": True}, paper_bgcolor = colors["lightblack"], plot_bgcolor=colors["lightblack"], font = {'color': colors["white"]})
    fig1.update_xaxes(showline=True, linewidth=1, linecolor=colors['darkgreyblue'], gridcolor=colors['darkgreyblue'])
    fig1.update_yaxes(showline=True, linewidth=1, linecolor=colors['darkgreyblue'], gridcolor=colors['darkgreyblue'])
    
    row = len(df)-1
    specdata1=df.iloc[row, 0]
    specdata2=df.iloc[row, 1]
  
    fig2= go.Figure(go.Indicator(
    mode = "gauge+number",
    value = specdata1,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Pažnja"},
    gauge = {'axis': {'range': [None, 100], 'tickcolor': colors['white']},
    'bar': {'color': colors["lightblue"]},
    'bgcolor': colors["lightblack"],
    'bordercolor': colors["lightblack"]
    },
    ))
    fig3 = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = specdata2,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Meditacija"},
    gauge = {'axis': {'range': [None, 100], 'tickcolor': colors['white']},
    'bar': {'color': colors["lightblue"]},
    'bgcolor': colors["lightblack"],
    'bordercolor': colors["lightblack"]
    },
    ))
    fig2.update_layout({"height": 280,"autosize":False}, paper_bgcolor = colors["lightblack"], font = {'color': colors["white"]})
    fig3.update_layout({"height": 280,"autosize":False}, paper_bgcolor = colors["lightblack"], font = {'color': colors["white"]})
    return [fig1, fig2, fig3 ]

#picture upload
@app.callback(Output('output-image-upload', 'children'),
              Input("btn-forward", "n_clicks"),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))
def update_output(n, list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        if n is None:
            return children[0]
        else:
            if len(list_of_contents)-1 < n:
                n=0
            return children[n]

new_data=[]
#start interval - recording waves
@app.callback([Output('interval-component2', 'n_intervals'), 
                Output('interval-component2', 'disabled')],
                Input('btn-play','n_clicks'))
def start_count(n_play):
    if n_play is None:
        return
    elif n_play!=0 and n_play!=None and n_play%2!=0:
        return 0, False
    else:
        new_data.clear()
        return 0, True

#record waves
@app.callback(Output('output-csv','children'), Input('interval-component2', 'n_intervals'))
def updateCsv(n_intervals):
        input_csv=pd.read_csv('data.csv')
        sub_array=[]
        row = len(input_csv)-1
        spec_dat=input_csv.iloc[row]
        sub_array.append(spec_dat["attention"])
        sub_array.append(spec_dat["meditation"])
        sub_array.append(spec_dat["delta"])
        sub_array.append(spec_dat["theta"])
        sub_array.append(spec_dat["lowalpha"])
        sub_array.append(spec_dat["highalpha"])
        sub_array.append(spec_dat["lowbeta"])
        sub_array.append(spec_dat["highbeta"])
        sub_array.append(spec_dat["lowgamma"])
        sub_array.append(spec_dat["highgamma"])
        new_data.append(sub_array)
        return  html.Div("Broj zapisa: {}".format(n_intervals), style={'color': colors['white'], 'fontSize':20})

#add recordings to csv
@app.callback(Output('download-csv', 'children'),
            [Input('btn-download','n_clicks'), Input('radioitems-input', 'value')])
def save_csv(n_download, value):
    if n_download is None:
        return
    if n_download>0 and n_download is not None:
        fieldnames = ["attention", "meditation", "delta", "theta", "lowalpha", "highalpha", "lowbeta", "highbeta", "lowgamma", "highgamma", "state" ]
        with open('train_data.csv', 'a', newline='') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            state=2
            if value=='pozitivno':
                state=2
            elif value=='negativno':
                state=0
            else:
                state=1
            for row in range(len(new_data)):
                info={
                    "attention": new_data[row][0],
                    "meditation": new_data[row][1],
                    "delta": new_data[row][2], 
                    "theta": new_data[row][3], 
                    "lowalpha": new_data[row][4], 
                    "highalpha": new_data[row][5], 
                    "lowbeta": new_data[row][6], 
                    "highbeta": new_data[row][7], 
                    "lowgamma": new_data[row][8], 
                    "highgamma": new_data[row][9],
                    "state": state
                }
                csv_writer.writerow(info)
            new_data.clear()            

#open-close modal
@app.callback(
    Output("modal", "is_open"),
    [Input("btn-model", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

#connect to mindwave
@app.callback(Output("connect-output", "children"), Input("btn-connect", "n_clicks"))
def connect(n):
    if n is None or n == 0:
        return
    elif n!=0 and n is not None:
        reader.start()
    else:
        return

#disconnect mindwave - restart app
@app.callback(Output("disconnect-output", "children"), Input("btn-disconnect", "n_clicks"))
def disconnect(n):
    if n is None or n == 0:
        return
    else:
        reader.stop()
        restart()

#start interval for prediction
@app.callback([Output('interval-component3', 'n_intervals'), 
                Output('interval-component3', 'disabled')],
                Input('btn-start-prediction','n_clicks'))
def start_stop_prediction(n):
    if n is None or n == 0:
        return [0, True]
    elif n!=0 and n is not None and n%2!=0:
        return [0, False]
    else:
        return [0, True]

#confusion matrix
def confusion_matrix_plot(confusion_matrix):
    # categories=['Negativno','Neutralno, 'Pozitivno']
    categories=['Negativno', 'Pozitivno']
    fig= px.imshow(confusion_matrix, 
                    labels=dict(x="Stvarne oznake", y="Predviđene oznake"),
                    aspect="auto",
                    x=categories,
                    y=categories,
                    text_auto=True,
                    title="Matrica konfuzije",
                    color_continuous_scale="Teal",
                )
    fig.update_layout(paper_bgcolor = colors["lightblack"], plot_bgcolor=colors["lightblack"], font = {'color': colors["white"]})
    fig.update_xaxes(showline=False) 
    fig.update_yaxes(showline=False)
    return fig
#accuracy
def accuracy_k_plot(neighbors, train_accuracy, test_accuracy):
    mode = 'lines'
    fig = px.scatter(title="Točnost modela")
    fig.add_scatter(name='Testiranje', x=neighbors, y=test_accuracy, mode=mode)
    fig.add_scatter(name='Treniranje', x=neighbors, y=train_accuracy, mode=mode)
    fig.update_layout(yaxis={"title": "Točnost"},xaxis={"title":"Broj susjeda"},paper_bgcolor = colors["lightblack"], plot_bgcolor=colors["lightblack"], font = {'color': colors["white"]})
    fig.update_xaxes(showline=True, linewidth=1, linecolor=colors['darkgreyblue'], gridcolor=colors['darkgreyblue'])
    fig.update_yaxes(showline=True, linewidth=1, linecolor=colors['darkgreyblue'], gridcolor=colors['darkgreyblue'])
    return fig

@app.callback([Output('knn-confusion','figure'), Output('knn-accuracy','figure')],
                 [Input('slider_k', 'value'), Input('btn-change-k', 'n_clicks')])
def update_model(value, n):
    if n is None or n == 0:
        return [confusion_matrix_plot(model.c_matrix), accuracy_k_plot(model.neighbors, model.train_accuracy, model.test_accuracy)] 
    elif n!=0 and n is not None:
        model.knn_model, model.c_matrix, model.neighbors, model.train_accuracy, model.test_accuracy = model.model(model.train_data, value)
        return [confusion_matrix_plot(model.c_matrix), accuracy_k_plot(model.neighbors, model.train_accuracy, model.test_accuracy)]
    else:
        return [confusion_matrix_plot(model.c_matrix), accuracy_k_plot(model.neighbors, model.train_accuracy, model.test_accuracy)]

#predict
@app.callback([Output('happy','style'),Output('sad','style')],
                Input('interval-component3', 'n_intervals'))
def predict(n):
    df = pd.read_csv('data.csv')
    row = len(df)-1
    specdata=[df.iloc[row, 3:10]]
    prediction, predict_proba = model.predict(model.knn_model,specdata)
    pred_proba_split = np.array_split(predict_proba[0], 3)
    # if prediction == 0:
    #       return [{'color': colors["shadow"], "fontSize": 46},{'color': colors["shadow"], "fontSize": 46},{'color': colors["lightblue"], "fontSize": 46}]
    # elif prediction == 2:
    #     return [{'color': colors["lightblue"], "fontSize": 46},{'color': colors["shadow"], "fontSize": 46},{'color': colors["shadow"], "fontSize": 46}]
    # else:
    #     return [{'color': colors["shadow"], "fontSize": 46},{'color': colors["lightblue"], "fontSize": 46},{'color': colors["shadow"], "fontSize": 46}]
    if prediction == 0:
          return [{'color': colors["shadow"], "fontSize": 69},{'color': colors["lightblue"], "fontSize": 69}]
    else:
        return [{'color': colors["lightblue"], "fontSize": 69},{'color': colors["shadow"], "fontSize": 69}]

if __name__ == '__main__':
    app.run_server(debug=True)


# def write_csv(name):
#     fieldnames = ["attention", "meditation", "delta", "theta", "lowalpha", "highalpha", "lowbeta", "highbeta", "lowgamma", "highgamma", "state" ]
#     with open(name, 'w', newline='') as csv_file:
#         csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#         csv_writer.writeheader()

# write_csv('train_data.csv')
