import json
import math
import pandas as pd
import numpy as np

#Plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

#Dash
from dash import Dash
from dash.dependencies import Input, Output
import dash.dependencies
from dash import dcc
from dash import html

dash.register_page(__name__)
#-------------------#
#------Options------#
#-------------------#
pd.options.plotting.backend = "plotly"
#external_stylesheets = ['https://bootswatch.com/5/flatly/bootstrap.min.css'] #https://bootswatch.com/5/morph/bootstrap.min.css
pio.templates.default = "seaborn"#plotly_dark


#-------------------#
#---Read Functions--#
#-------------------#

# functions
def load_tsv(path):
    df = pd.read_csv(path,sep='\t');
    # dframe.drop(dframe[dframe.type.str.contains("_ex")].index, inplace=True) -> TO Drop specific lines!
    return df
def load_csv(path):
    df = pd.read_csv(path,sep=';');
    return df

def read_data(path):
    df = load_tsv(path)
    
    # normalize timestamp to start at 0
    firstTimestamp = df["timestamp"].iloc[0]
    df["timestamp"] = df.apply(lambda x: x["timestamp"] - firstTimestamp,axis=1)
    
    # Timestamps from miliseconds to seconds
    df["timestamp"] = df.apply(lambda x: x["timestamp"] / 1000,axis=1)
    
    #Timestamps as index
    df.set_index('timestamp',inplace=True)
    
    #Cutting 0 FPS columns because irrelevant
#    df = df.loc[df['fps'] > 0]
    
#    df = df.apply(pd.to_numeric, errors='coerce')
    #rework boolean for plotting
    df.replace({False: 0, True: 1}, inplace=True)
    return df
def add_conditions(subject,fig,propertyName):
    shapes = []
    intersections = subjects[subject][["intersection"]]
    #mark the rows where the content changes
    changes = intersections["intersection"].shift() != intersections["intersection"]
    #get the rows where the change happens
    changeRows = intersections.loc[changes[changes].index, :]
    lastElem = intersections.iloc[0]

    #create axspan for each
    for index, row in changeRows.iterrows():
        if index > 0:
            condition = conditions.loc[conditions["intersection"] == lastElem[1]["intersection"]]    
            #Colors..
            if lastElem[1]["intersection"] == "NONE":
                lastElem = [index,row]
                continue
            elif  condition.empty:
                color = COLOR_SIT_X_HEX
            elif  condition["condition"].values[0] == 'x':
                color = COLOR_SIT_X_HEX
            elif  condition["condition"].values[0] == 'a':
                color = COLOR_SIT_A_HEX
            elif  condition["condition"].values[0] == 'b':
                color = COLOR_SIT_B_HEX
            elif  condition["condition"].values[0] == 'c':
                color = COLOR_SIT_C_HEX
            elif  condition["condition"].values[0] == 'd':
                color = COLOR_SIT_D_HEX
            elif  condition["condition"].values[0] == 'e':
                color = COLOR_SIT_E_HEX
            shapes.append({
                'type': 'rect',
    #            xref: 'x',
    #            yref: 'paper',
                'x0': lastElem[0],
                'y0': subjects[subject][propertyName].min(),
                'x1': index,
                'y1': subjects[subject][propertyName].max(),
                'fillcolor': color,
                'opacity': 0.5,
                'line': {
                    'width': 0
                }
            })
        lastElem = [index,row]
    fig.update_layout(shapes = shapes)
def getSiutationTimes(subject):
    intersections = subjects[subject][["intersection"]]
    #mark the rows where the content changes
    changes = intersections["intersection"].shift() != intersections["intersection"]
    #get the rows where the change happens
    changeRows = intersections.loc[changes[changes].index, :]
    lastElem = intersections.iloc[0]

    times = {
        "none" : [],
        "x" : [],
        "a" : [],
        "b" : [],
        "c" : [],
        "d" : [],
        "e" : []
    }
    #create axspan for each
    for index, row in changeRows.iterrows():
        if index > 0:
            condition = conditions.loc[conditions["intersection"] == lastElem[1]["intersection"]]    
            if lastElem[1]["intersection"] == "NONE":
                times["none"].append(index - lastElem[0])
            elif  condition.empty:
                times["none"].append(index - lastElem[0])
            elif  condition["condition"].values[0] == 'x':
                times["x"].append(index - lastElem[0])
            elif  condition["condition"].values[0] == 'a':
                times["a"].append(index - lastElem[0])
            elif  condition["condition"].values[0] == 'b':
                times["b"].append(index - lastElem[0])
            elif  condition["condition"].values[0] == 'c':
                times["c"].append(index - lastElem[0])
            elif  condition["condition"].values[0] == 'd':
                times["d"].append(index - lastElem[0])
            elif  condition["condition"].values[0] == 'e':
                times["e"].append(index - lastElem[0])
        lastElem = [index,row]
    return times

#-------------------#
#------Data Read----#
#-------------------#
#Global read conditions
conditions = load_csv("./logs/vr/conditions.csv")

df_subject1 = read_data('./logs/vr/subject1_2022-06-28_103935_Condition_0.tsv')
df_subject2 = read_data('./logs/vr/subject2_2022-06-28_152542_Condition_0.tsv')
df_subject3 = read_data('./logs/vr/subject3_2022-06-28_164830_Condition_0.tsv')
df_subject4 = read_data('./logs/vr/subject4_2022-06-29_134215_Condition_0.tsv')
df_subject5 = read_data('./logs/vr/subject5_2022-06-29_143922_Condition_0.tsv')
df_subject6 = read_data('./logs/vr/subject6_2022-06-29_153217_Condition_0.tsv')
df_subject7 = read_data('./logs/vr/subject7_2022-06-30_133951_Condition_0.tsv')
df_subject8= read_data('./logs/vr/subject8_2022-07-05_161117_Condition_0.tsv')
df_subject9= read_data('./logs/vr/subject9_2022-07-05_161117_Condition_0.tsv')
df_subject10 = read_data('./logs/vr/subject10_2022-07-05_175819_Condition_0.tsv')
df_subject11 = read_data('./logs/vr/subject11_2022-07-06_134238_Condition_0.tsv')
df_subject12 = read_data('./logs/vr/subject12_2022-07-08_111356_Condition_0.tsv')

subjects = {
    "Subject 1" : df_subject1,
    "Subject 2" : df_subject2,
    "Subject 3" : df_subject3,
    "Subject 4" : df_subject4,
    "Subject 5" : df_subject5,
    "Subject 6" : df_subject6,
    "Subject 7" : df_subject7,
    "Subject 8" : df_subject8,
    "Subject 9" : df_subject9,
    "Subject 10" : df_subject10,
    "Subject 11" : df_subject11,
    "Subject 12" : df_subject12
}
available_data_labels = list(subjects.keys())

#-------------------#
#-----App Layout----#
#-------------------#
app.layout = html.Div([
    html.Div([dcc.Markdown('''### VR Study''',id='headline_subject')],style={'width': '100%', 'display': 'block'}),
    html.Div([
        html.Label('Subject'),
        dcc.Dropdown(
            id='filter_subject',
            options=[{'label': i, 'value': i} for i in available_data_labels],
            value='Choose Subject '
    )],style={'width': '100%', 'display': 'inline-block'}),
    html.Div([
        html.Label('Timestamp (Seconds)'),
        dcc.Slider(
            id='filter-slider-timestamp',
            min=0,
            max=0,
            value=0,
            tooltip={"placement": "bottom", "always_visible": True},
            step=60
        )
    ], style={'width': '100%', 'display': 'inline-block'}),
    html.Div([
        html.Label('NO Condition',style={'backgroundColor' : COLOR_SIT_X_HEX,'color': "#FFFFFF",'borderRadius':'5px','marginLeft':'10px','padding': '10px'}),
        html.Label('Condition A',style={'backgroundColor' : COLOR_SIT_A_HEX,'color': "#FFFFFF",'borderRadius':'5px','marginLeft':'10px','padding': '10px'}),
        html.Label('Condition B',style={'backgroundColor' : COLOR_SIT_B_HEX,'color': "#FFFFFF",'borderRadius':'5px','marginLeft':'10px','padding': '10px'}),
        html.Label('Condition C',style={'backgroundColor' : COLOR_SIT_C_HEX,'color': "#FFFFFF",'borderRadius':'5px','marginLeft':'10px','padding': '10px'}),
        html.Label('Condition D',style={'backgroundColor' : COLOR_SIT_D_HEX,'color': "#FFFFFF",'borderRadius':'5px','marginLeft':'10px','padding': '10px'}),
        html.Label('Condition E',style={'backgroundColor' : COLOR_SIT_E_HEX,'color': "#FFFFFF",'borderRadius':'5px','marginLeft':'10px','padding': '10px'}),
    ], style={'width': '100%', 'display': 'inline-block'}),
    html.Div([dcc.Graph(id='headmovement-scatter',hoverData={'points': [{'x': 0}]})], style={'width': '100%', 'display': 'block'}),
    html.Div([dash_table.DataTable(
        id='collision_table',
        columns=[{"name": "Situation", "id": "sit", "presentation": "markdown"},
                 {"name": "Collisions", "id": "colisions", "presentation": "input"},
                 {"name": "Head moved left", "id": "leftHead", "presentation": "markdown"},
                 {"name": "Head moved right", "id": "rightHead", "presentation": "markdown"}],
        data=[{"sit": 'None'},{"sit": 'X'},{"sit": 'A'},{"sit": 'B'},{"sit": 'C'},{"sit": 'D'},{"sit": 'E'}],
    )], style={'width': '45%', 'display': 'inline-block'}),
        html.Div([dash_table.DataTable(
        id='times_table',
        columns=[{"name": "Situation", "id": "sit", "presentation": "markdown"},
                 {"name": "Time 1", "id": "t_1", "presentation": "input"},
                 {"name": "Time 2", "id": "t_2", "presentation": "input"},
                 {"name": "Time 3", "id": "t_3", "presentation": "input"},
                 {"name": "average", "id": "average", "presentation": "input"}],
        data=[{"sit": 'None'},{"sit": 'X'},{"sit": 'A'},{"sit": 'B'},{"sit": 'C'},{"sit": 'D'},{"sit": 'E'}],
    )], style={'width': '45%', 'display': 'inline-block'}),
    html.Div([dcc.Graph(id='rolls-scatter',hoverData={'points': [{'x': 0}]})], style={'width': '100%', 'display': 'block'}),
    html.Div([dcc.Graph(id='pitchs-scatter',hoverData={'points': [{'x': 0}]})], style={'width': '100%', 'display': 'block'}),
    html.Div([dcc.Graph(id='velocity')], style={'display': 'inline-block', 'width': '50%'}),
    html.Div([dcc.Graph(id='acceleration')], style={'display': 'inline-block', 'width': '50%'})
])
#-------------------#
#---App Callbacks---#
#-------------------#

# Headline
@app.callback(
    dash.dependencies.Output('headline_subject', 'children'),
    [dash.dependencies.Input('filter_subject', 'value')])
def update_headline(subject):
    if not subject in subjects:
        return "### VR Study"
    return "### VR Study : " + subject

#Crashes
@app.callback(
    dash.dependencies.Output('collision_table', 'data'),
    [dash.dependencies.Input('filter_subject', 'value')])
def update_headline(subject):
    if not subject in subjects:
        return
    crashCount = {
        "none" : 0,
        "x" : 0,
        "a" : 0,
        "b" : 0,
        "c" : 0,
        "d" : 0,
        "e" : 0
    }
    crashes = subjects[subject][["collisions","intersection"]].loc[subjects[subject]["collisions"] != 0]   
    for index,row in crashes.iterrows():
        if row["intersection"] in conditions["intersection"].unique():
            condition = conditions.loc[conditions["intersection"] == row["intersection"]]["condition"].iloc[0]
            crashCount[condition] = crashCount[condition] + 1
    return [{"sit": 'None',"colisions" : crashCount["none"]},
            {"sit": 'X',"colisions" : crashCount["x"]},
            {"sit": 'A',"colisions" : crashCount["a"]},
            {"sit": 'B',"colisions" : crashCount["b"]},
            {"sit": 'C',"colisions" : crashCount["c"]},
            {"sit": 'D',"colisions" : crashCount["d"]},
            {"sit": 'E',"colisions" : crashCount["e"]}]
#Situation Times
@app.callback(
    dash.dependencies.Output('times_table', 'data'),
    [dash.dependencies.Input('filter_subject', 'value')])
def update_headline(subject):
    if not subject in subjects:
        return
    times = getSiutationTimes(subject)
    return [{"sit": 'None'},
            {"sit": 'X'},
            {"sit": 'A',"t_1" : times["a"][0],"t_2" : times["a"][1],"t_3" : times["a"][2],"average" : (times["a"][0] + times["a"][1] + times["a"][2]) / 3},
            {"sit": 'B',"t_1" : times["b"][0],"t_2" : times["b"][1],"t_3" : times["b"][2],"average" : (times["b"][0] + times["b"][1] + times["b"][2]) / 3},
            {"sit": 'C',"t_1" : times["c"][0],"t_2" : times["c"][1],"t_3" : times["c"][2],"average" : (times["c"][0] + times["c"][1] + times["c"][2]) / 3},
            {"sit": 'D',"t_1" : times["d"][0],"average" : times["d"][0]},
            {"sit": 'E',"t_1" : times["e"][0],"t_2" : times["e"][1],"t_3" : times["e"][2],"average" : (times["e"][0] + times["e"][1] + times["e"][2]) / 3}]

#Timestamps
@app.callback(
    dash.dependencies.Output('filter-slider-timestamp', 'value'),
    [dash.dependencies.Input('filter_subject', 'value'),
    dash.dependencies.Input('headmovement-scatter', 'hoverData')])
def update_timestampValue(subject,hoverdata):
    if not subject in subjects:
        return 0
    
    lastTimestamp = float(subjects[subject].iloc[-1].name)
    #display("The experiment from subject "+infos["subject"]+" lasted " + str(lastTimestamp) + " seconds and started at " + infos["starttime"])
    halfminuteSteps = np.arange(0,lastTimestamp,30)
    halfminuteStepLabels = list(map(lambda x: str(int(x)), halfminuteSteps))
    if "x" in hoverdata['points'][0]:
        return hoverdata['points'][0]["x"]
    else:
        return 0
@app.callback(
    dash.dependencies.Output('filter-slider-timestamp', 'min'),
    [dash.dependencies.Input('filter_subject', 'value')])
def update_timestampMin(subject):
    if not subject in subjects:
        return 0
    return subjects[subject].index.min()
@app.callback(
    dash.dependencies.Output('filter-slider-timestamp', 'max'),
    [dash.dependencies.Input('filter_subject', 'value')])
def update_timestampMax(subject):
    if not subject in subjects:
        return 0
    return subjects[subject].index.max()

#Headmovement
@app.callback(
    dash.dependencies.Output('headmovement-scatter', 'figure'),
    [dash.dependencies.Input('filter_subject', 'value'),
     dash.dependencies.Input('filter-slider-timestamp', 'value')])
def update_headrotation(subject, timestamp):
    if not subject in subjects:
        return go.Figure()
    fig = go.Figure()
#    fig.add_trace(go.Scatter(
#        x=subjects[subject].index,
#        y=subjects[subject]["hmdHeadRotationY"],
#        mode="lines",
#        name="Headrotation (Yaw)")
#    )
    fig.add_trace(go.Scatter(
        x=subjects[subject].index,
        y=subjects[subject]["hmdRotationY"],
        mode="lines",
        name="Headrotation (relative Yaw)")
    )
    fig.add_annotation(
        x=timestamp, 
        y=subjects[subject]["hmdRotationY"].max(),
        text="Chosen Time " + str(timestamp),
        showarrow=False,
        yshift=10
    )
    fig.update_layout(
        title_text="Headmovement",
        autosize=True,
        hovermode='closest',
        shapes = [
            # Line Horizontal
            {
                'type': 'line',
                'x0': timestamp,
                'y0': subjects[subject]["hmdRotationY"].min(),
                'x1': timestamp,
                'y1': subjects[subject]["hmdRotationY"].max(),
                'line': {
                    'color': 'rgb(50, 171, 96)',
                    'width': 2
                },
            }
        ]
    )
    add_conditions(subject,fig,"hmdRotationY")
    fig.update_xaxes(title_text='Time(seconds)')
    fig.update_yaxes(title_text='Headrotation (Degrees)')

    return fig

#Roll
@app.callback(
    dash.dependencies.Output('rolls-scatter', 'figure'),
    [dash.dependencies.Input('filter_subject', 'value'),
     dash.dependencies.Input('filter-slider-timestamp', 'value')])
def update_rolls(subject, timestamp):
    if not subject in subjects:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=subjects[subject].index,
        y=subjects[subject]["hmdRotationZ"],
        mode="lines",
        name="HmdRotationZ")
    )
    fig.add_annotation(
        x=timestamp, 
        y=subjects[subject]["hmdRotationZ"].max(),
        text="Chosen Time " + str(timestamp),
        showarrow=False,
        yshift=10
    )
    fig.update_layout(
        title_text="Roll (HMDRotationZ)",
        autosize=True,
        hovermode='closest',
        shapes = [
            # Line Horizontal
            {
                'type': 'line',
                'x0': timestamp,
                'y0': subjects[subject]["hmdRotationZ"].min(),
                'x1': timestamp,
                'y1': subjects[subject]["hmdRotationZ"].max(),
                'line': {
                    'color': 'rgb(50, 171, 96)',
                    'width': 2
                },
            }
        ]
    )
    add_conditions(subject,fig,"hmdRotationZ")
    fig.update_xaxes(title_text='Time(seconds)')
    fig.update_yaxes(title_text='Roll (Degrees)')

    return fig
#Pitch
@app.callback(
    dash.dependencies.Output('pitchs-scatter', 'figure'),
    [dash.dependencies.Input('filter_subject', 'value'),
     dash.dependencies.Input('filter-slider-timestamp', 'value')])
def update_pitchs(subject, timestamp):
    if not subject in subjects:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=subjects[subject].index,
        y=subjects[subject]["hmdRotationX"],
        mode="lines",
        name="HmdRotationX")
    )
    fig.add_annotation(
        x=timestamp, 
        y=subjects[subject]["hmdRotationX"].max(),
        text="Chosen Time " + str(timestamp),
        showarrow=False,
        yshift=10
    )
    fig.update_layout(
        title_text="Pitch (HmdRotationX)",
        autosize=True,
        hovermode='closest',
        shapes = [
            # Line Horizontal
            {
                'type': 'line',
                'x0': timestamp,
                'y0': subjects[subject]["hmdRotationX"].min(),
                'x1': timestamp,
                'y1': subjects[subject]["hmdRotationX"].max(),
                'line': {
                    'color': 'rgb(50, 171, 96)',
                    'width': 2
                },
            }
        ]
    )
    add_conditions(subject,fig,"hmdRotationX")
    fig.update_xaxes(title_text='Time(seconds)')
    fig.update_yaxes(title_text='Pitch (Degrees)')

    return fig
#Acceleration
@app.callback(
    dash.dependencies.Output('acceleration', 'figure'),
    [dash.dependencies.Input('filter_subject', 'value'),
     dash.dependencies.Input('filter-slider-timestamp', 'value')])
def update_acceleration(subject, timestamp):
    if not subject in subjects:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=subjects[subject].index,
        y=subjects[subject]["speed"],
        mode="lines")
    )
    fig.add_annotation(
        x=timestamp, 
        y=subjects[subject]["speed"].max(),
        text="Chosen Time " + str(timestamp),
        showarrow=False,
        yshift=10
    )
    fig.update_layout(
        title_text="Acceleration (speed)",
        autosize=True,
        hovermode='closest',
        shapes = [
        # Line Horizontal
        {
            'type': 'line',
            'x0': timestamp,
            'y0': subjects[subject]["speed"].min(),
            'x1': timestamp,
            'y1': subjects[subject]["speed"].max(),
            'line': {
                'color': 'rgb(50, 171, 96)',
                'width': 2
            },
        }]
    )
    add_conditions(subject,fig,"speed")
    fig.update_xaxes(title_text='Time (seconds)')
    fig.update_yaxes(title_text='speed')
    return fig
#Velocity
@app.callback(
    dash.dependencies.Output('velocity', 'figure'),
    [dash.dependencies.Input('filter_subject', 'value'),
     dash.dependencies.Input('filter-slider-timestamp', 'value')])
def update_velocity(subject, timestamp):
    if not subject in subjects:
        return go.Figure()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=subjects[subject].index,
        y=subjects[subject]["velocityY"],
        mode="lines")
    )
    fig.add_annotation(
        x=timestamp, 
        y=subjects[subject]["velocityY"].max(),
        text="Chosen Time " + str(timestamp),
        showarrow=False,
        yshift=10
    )
    fig.update_layout(
        title_text="Velocity",
        autosize=True,
        hovermode='closest',
        shapes = [
        # Line Horizontal
        {
            'type': 'line',
            'x0': timestamp,
            'y0': subjects[subject]["velocityY"].min(),
            'x1': timestamp,
            'y1': subjects[subject]["velocityY"].max(),
            'line': {
                'color': 'rgb(50, 171, 96)',
                'width': 2
            },
        }]
    )
    add_conditions(subject,fig,"velocityY")
    fig.update_xaxes(title_text='Time (seconds)')
    fig.update_yaxes(title_text='Velocity Y (km/h)')

    return fig
#-------------------#
#------Server-------#
#-------------------#

if __name__ == '__main__':
	app.run_server(debug=True)