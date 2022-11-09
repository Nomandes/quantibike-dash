#import dash
#from dash import html, dcc
#dash.register_page(__name__)
#layout = html.Div(children=[
#    html.H1(children='FIELD'),
#
#    html.Div(children='''
#        This is our Archive page content.
#    '''),
#
#])
import dash
from dash import Input, Output, html, dcc

import json
import math
import pandas as pd
import numpy as np
import pysrt

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

dash.register_page(__name__)

pd.options.plotting.backend = "plotly"
pio.templates.default = "simple_white"#"seaborn"#plotly_dark

COLOR_SIT_A_HEX = "#7a5b7b"
COLOR_SIT_B_HEX = "#f9dbbd"
COLOR_SIT_C_HEX = "#fca17d"
COLOR_SIT_D_HEX = "#68534d"
COLOR_SIT_E_HEX = "#5b7b7a"
COLOR_SIT_X_HEX = "#8f2d41"

########
#FUNCTIONS
########

def load_csv(path):
    df = pd.read_csv(path,sep=';');
    return df
def add_conditions(subject,fig,propertyName):
    subs = srts[subject]
    shapes = []
    for sub in subs:
        # If Log was writte before video, add the delay to the times to get correct times
        if srt_delays[subject] < 0:
            start = sub.start.minutes * 60 + sub.start.seconds + abs(srt_delays[subject])
            end = sub.end.minutes * 60 + sub.end.seconds + abs(srt_delays[subject])
        else:
            start = sub.start.minutes * 60 + sub.start.seconds - abs(srt_delays[subject])
            end = sub.end.minutes * 60 + sub.end.seconds - abs(srt_delays[subject])
        #display("Condition lasted from {} to {}".format(start,end))#
        color = "#FFFFFF"
        if "Condition Start" in sub.text:
            color = COLOR_SIT_X_HEX
        elif "Condition X" in sub.text:
            color = COLOR_SIT_X_HEX
        elif "Condition A" in sub.text:
            color = COLOR_SIT_A_HEX
        elif "Condition B" in sub.text:
            color = COLOR_SIT_B_HEX
        elif "Condition C" in sub.text:
            color = COLOR_SIT_C_HEX
        elif "Condition D" in sub.text:
            color = COLOR_SIT_D_HEX
        elif "Condition E" in sub.text:
            color = COLOR_SIT_E_HEX
        else:
            color = COLOR_SIT_X_HEX
        shapes.append({
            'type': 'rect',
#            xref: 'x',
#            yref: 'paper',
            'x0': start,
            'y0': subjects[subject][propertyName].min(),
            'x1': end,
            'y1': subjects[subject][propertyName].max(),
            'fillcolor': color,
            'opacity': 0.5,
            'line': {
                'width': 0
            }
        })
    fig.update_layout(shapes = shapes)
def include_euler_angles(df):
    # Rotation Matrix Phone and Airpods
    mRotPhone = []
    mRotAir = []
    for index, row in df.iterrows():
        rotPhone_first = row[["phoneMotionData.rotationMatrix.m1.1","phoneMotionData.rotationMatrix.m1.2","phoneMotionData.rotationMatrix.m1.3",]]
        rotPhone_second = row[["phoneMotionData.rotationMatrix.m2.1","phoneMotionData.rotationMatrix.m2.2","phoneMotionData.rotationMatrix.m2.3"]]
        rotPhone_third =  row[["phoneMotionData.rotationMatrix.m3.1","phoneMotionData.rotationMatrix.m3.2","phoneMotionData.rotationMatrix.m3.3"]]
        rotPhone_nump = np.array([rotPhone_first.to_numpy(),rotPhone_second.to_numpy(),rotPhone_third.to_numpy()])
        mRotPhone.append(rotPhone_nump)

        rotAir_first = row[["airpodMotionData.rotationMatrix.m1.1","airpodMotionData.rotationMatrix.m1.2","airpodMotionData.rotationMatrix.m1.3",]]
        rotAir_second = row[["airpodMotionData.rotationMatrix.m2.1","airpodMotionData.rotationMatrix.m2.2","airpodMotionData.rotationMatrix.m2.3"]]
        rotAir_third =  row[["airpodMotionData.rotationMatrix.m3.1","airpodMotionData.rotationMatrix.m3.2","airpodMotionData.rotationMatrix.m3.3"]]
        rotAir_nump = np.array([rotAir_first.to_numpy(),rotAir_second.to_numpy(),rotAir_third.to_numpy()])
        mRotAir.append(rotAir_nump)

    # Relative Rotations
    mRotRelatives = []
    for index, rotPhone in enumerate(mRotPhone):
        mRotRelatives.append(rotPhone.dot(mRotAir[index].transpose()))

    # Yaw Values
    relativeYaws = []
    relativeRoll = []
    relativePitch = []
    for index, rotRelative in enumerate(mRotRelatives):
        relativeYaws.append(math.atan2(rotRelative[0,1],rotRelative[0,0])) #Indexes shifted because starting at 0
        relativeRoll.append(math.atan2(rotRelative[1,2],rotRelative[2,2])) #Indexes shifted because starting at 0
        relativePitch.append(- math.asin(rotRelative[0,2]))                #Indexes shifted because starting at 0

    #Convert Radians to Degree
    realtiveYawsDegree = np.degrees(relativeYaws)
    relativeRollDegree = np.degrees(relativeRoll)
    relativePitchDegree = np.degrees(relativePitch)

    #import into frame
    df["relativeYaw"] = realtiveYawsDegree.tolist()
    df["relativeRoll"] = relativeRollDegree.tolist()
    df["relativePitch"] = relativePitchDegree.tolist()

def read_data(url):
    with open(url,'r') as f:
        data = json.loads(f.read())

    #get infos and delete theme from json
    #infos = data.pop("infos")
    #df["motionTimestampDiff"] = df.apply(lambda x: x["airpodMotionData.timestamp"] - x["phoneMotionData.timestamp"],axis=1)

    # normalize json to readable frame
    df = pd.json_normalize(data,record_path=["timestamps"])

    # convert objects to floats
    df = df.apply(pd.to_numeric, errors='coerce')

    #set timestamp as index
    df.set_index('timestamp',inplace=True)

    #Velocity m/s to km/h
    if 'locationData.velocity' in df.columns:
        df["locationData.velocity"] = df["locationData.velocity"].apply(lambda x: x * 3.6)
    #calculate Headrotation
    df["headRotY"] = df.apply(lambda x: x["airpodMotionData.yaw"] - x["phoneMotionData.yaw"],axis=1)
    df["headRotY"] = np.degrees(df["headRotY"]).tolist()
    df["airpodsYawDegrees"] = np.degrees(df["airpodMotionData.yaw"]).tolist()
    df["phoneYawDegrees"] = np.degrees(df["phoneMotionData.yaw"]).tolist()

    df["airpodsRollDegrees"] = np.degrees(df["airpodMotionData.roll"]).tolist()
    df["phoneRollDegrees"] = np.degrees(df["phoneMotionData.roll"]).tolist()

    df["airpodsPitchDegrees"] = np.degrees(df["airpodMotionData.pitch"]).tolist()
    df["phonePitchDegrees"] = np.degrees(df["phoneMotionData.pitch"]).tolist()
    include_euler_angles(df)
    return df

def read_srt(url):
    subs = pysrt.open(url)
    return subs

#SUBJECTS
df_subject1 = read_data("./logs/field/2022-10-17 124155-logfile-subject-1.json")
#df_subject2 = read_data("./logs/field/2022-10-20 111557-logfile-subject-2.json")
#df_subject3 = read_data("./logs/field/2022-10-25 124003-logfile-subject-3.json")
#df_subject4 = read_data("./logs/field/2022-10-25 142229-logfile-subject-4.json")
#df_subject5 = read_data("./logs/field/2022-10-27 165339-logfile-subject-5.json")
#df_subject6 = read_data("./logs/field/2022-10-28 101829-logfile-subject-6.json")
#df_subject7 = read_data("./logs/field/2022-10-28 154837-logfile-subject-7.json")
#Calculate Timestamp Diff for motionData

#SRTs
srt_subject1 = read_srt("./logs/field/srts/subs_participant_1.srt")
#srt_subject2 = read_srt("./logs/field/srts/subs_participant_2.srt")
#srt_subject3 = read_srt("/home/nomandes/quantibike//logs/participants/srts/subs_participant_3.srt")
#srt_subject4 = read_srt("./logs/field/srts/subs_participant_4.srt")
#srt_subject5 = read_srt("./logs/field/srts/subs_participant_5.srt")
#srt_subject6 = read_srt("./logs/field/srts/subs_participant_6.srt")
#srt_subject7 = read_srt("./logs/field/srts/subs_participant_7.srt")

# TO-DO!! -> Timestamp und steps dynamisch holen
lastTimestamp = float(df_subject1.iloc[-1].name)
#display("The experiment from subject "+infos["subject"]+" lasted " + str(lastTimestamp) + " seconds and started at " + infos["starttime"])
halfminuteSteps = np.arange(0,lastTimestamp,30)
halfminuteStepLabels = list(map(lambda x: str(int(x)), halfminuteSteps))

subjects = {
#    "DEBUG - Andrii Crazy" : df_andrii_crazy,
#    "TEST - Testride 1" : df_manu_realDeal,
#    "TEST - Testride 2" : df_manu_realDeal2,
#    "DEBUG - AirPods 3 Test" : df_airpods3,
#    "DEBUG - Airpods 3 - Uni" : df_air3_uni,
#    "DEBUG - Airpods 3 - Lichtwiese Hin": df_air3_lw_hin,
#    "DEBUG - Airpods 3 - Lichtwiese Zurück" : df_air3_lw_zurueck,
    "Subject 1" : df_subject1
    #"Subject 2" : df_subject2,
    #"Subject 3 - No Video" : df_subject3,
    #"Subject 4" : df_subject4,
    #"Subject 5" : df_subject5,
    #"Subject 6" : df_subject6,
    #"Subject 7" : df_subject7,
}
srts = {
    "Subject 1" : srt_subject1
    #"Subject 2" : srt_subject2,
#    "Subject 3" : srt_subject3,
    #"Subject 4" : srt_subject4,
    #"Subject 5" : srt_subject5,
    #"Subject 6" : srt_subject6,
    #"Subject 7" : srt_subject7
}
# If negative, the app was running before the camera. Little inconsistency in the study design
srt_delays = {
    "Subject 1" : - 8.0
    #"Subject 2" : - 29.0,
#    "Subject 3" : ,
   # "Subject 4" : -10.0,
    #"Subject 5" : 9.0,
    #"Subject 6" : -32.0,
    #"Subject 7" : 490.0
}
available_data_labels = list(subjects.keys())

#-------------------#
#-----App Layout----#
#-------------------#
layout = html.Div([
    html.Div([dcc.Markdown('''### Please Choose an Subject!''',id='headline_subject')],style={'width': '100%', 'display': 'block'}),
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
    html.Div([dcc.Graph(id='gps-coords')], style={'display': 'inline-block', 'width': '100%'}),
    html.Div([dcc.Graph(id='yaws-scatter',hoverData={'points': [{'x': 0}]})], style={'width': '100%', 'display': 'block'}),
    html.Div([dcc.Graph(id='rolls-scatter',hoverData={'points': [{'x': 0}]})], style={'width': '100%', 'display': 'block'}),
    html.Div([dcc.Graph(id='pitchs-scatter',hoverData={'points': [{'x': 0}]})], style={'width': '100%', 'display': 'block'}),
    html.Div([dcc.Graph(id='velocity')], style={'display': 'inline-block', 'width': '50%'}),
    html.Div([dcc.Graph(id='acceleration')], style={'display': 'inline-block', 'width': '50%'}),
    html.Div([dcc.Graph(id='altitude')], style={'display': 'inline-block', 'width': '50%'}),
    html.Div([dcc.Graph(id='phone_battery')], style={'display': 'inline-block', 'width': '50%'}),
    #html.Div([dcc.Graph(id='headMesh')], style={'display': 'inline-block', 'width': '25%'}),
])

#-------------------#
#---App Callbacks---#
#-------------------#

# Timestamps
@dash.callback(
    dash.dependencies.Output('filter-slider-timestamp', 'value'),
    [dash.dependencies.Input('filter_subject', 'value'),
    dash.dependencies.Input('headmovement-scatter', 'hoverData')])
def update_timestampValue(subject,hoverdata):
    if not subject in subjects:
        return 0
    # TO-DO!! -> Timestamp und steps dynamisch holen
    lastTimestamp = float(subjects[subject].iloc[-1].name)
    #display("The experiment from subject "+infos["subject"]+" lasted " + str(lastTimestamp) + " seconds and started at " + infos["starttime"])
    halfminuteSteps = np.arange(0,lastTimestamp,30)
    halfminuteStepLabels = list(map(lambda x: str(int(x)), halfminuteSteps))
    if "x" in hoverdata['points'][0]:
        return hoverdata['points'][0]["x"]
    else:
        return 0
@dash.callback(
    dash.dependencies.Output('filter-slider-timestamp', 'min'),
    [dash.dependencies.Input('filter_subject', 'value')])
def update_timestampMin(subject):
    if not subject in subjects:
        return 0
    return subjects[subject].index.min()
@dash.callback(
    dash.dependencies.Output('filter-slider-timestamp', 'max'),
    [dash.dependencies.Input('filter_subject', 'value')])
def update_timestampMax(subject):
    if not subject in subjects:
        return 0
    return subjects[subject].index.max()

#Headmovement
@dash.callback(
    dash.dependencies.Output('headmovement-scatter', 'figure'),
    [dash.dependencies.Input('filter_subject', 'value'),
     dash.dependencies.Input('filter-slider-timestamp', 'value')])
def update_headrotation(subject, timestamp):
    if not subject in subjects:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=subjects[subject].index,
        y=subjects[subject]["relativeYaw"],
        mode="lines",
        name="Headrotation (relative Yaw)")
    )
    fig.add_annotation(
        x=timestamp,
        y=subjects[subject]["relativeYaw"].max(),
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
                'y0': subjects[subject]["relativeYaw"].min(),
                'x1': timestamp,
                'y1': subjects[subject]["relativeYaw"].max(),
                'line': {
                    'color': 'rgb(50, 171, 96)',
                    'width': 2
                },
            }
        ]
    )
    add_conditions(subject,fig,"relativeYaw")
    fig.update_xaxes(title_text='Time(seconds)')
    fig.update_yaxes(title_text='Headrotation (Degrees)')

    return fig

#GPS Coordinates
@dash.callback(
    dash.dependencies.Output('gps-coords', 'figure'),
    [dash.dependencies.Input('filter_subject', 'value'),
     dash.dependencies.Input('filter-slider-timestamp', 'value')])
def update_gps_coords(subject, timestamp):
    if not subject in subjects:
        return go.Figure()

    fig = go.Figure()
    print("got timestamp " + str(timestamp))
    #Add all Points
    fig.add_trace(go.Scattermapbox(
        lon=subjects[subject]["locationData.longitude"],
        lat=subjects[subject]["locationData.latitude"],
        mode='lines',
        name="Track"
    ))

    #Highlight current point
    fig.add_trace(go.Scattermapbox(
        lon=[subjects[subject].iloc[[timestamp]]["locationData.longitude"]],
        lat=[subjects[subject].iloc[[timestamp]]["locationData.latitude"]],
        mode='markers',
        name="Position at Timestamp",
        marker=go.scattermapbox.Marker(
            size=12,
            color='rgb(50, 171, 96)',
            opacity=0.9,
            symbol= "bicycle"
        )
    ))

    fig.update_layout(
        title_text="GPS Coordinates",
        autosize=True,
        hovermode='closest',
        mapbox=dict(
            accesstoken="pk.eyJ1Ijoibm9tYW5kZXMiLCJhIjoiY2w3cmticXRhMGc0OTN3cDJ5Y2pqcnZ3YiJ9.R8snls4l3IE8Z2cgQ5Ti3g",
            #bearing=0,
            #pitch=0,
            center= dict(
                lat = subjects[subject].iloc[subjects[subject].shape[0]//2]["locationData.latitude"], #middle point of the log
                lon = subjects[subject].iloc[subjects[subject].shape[0]//2]["locationData.longitude"]
            ),
            zoom=13.5
    ))

    return fig
#Yaws
@dash.callback(
    dash.dependencies.Output('yaws-scatter', 'figure'),
    [dash.dependencies.Input('filter_subject', 'value'),
     dash.dependencies.Input('filter-slider-timestamp', 'value')])
def update_yaws(subject, timestamp):
    if not subject in subjects:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=subjects[subject].index,
        y=subjects[subject]["relativeYaw"],
        mode="lines",
        name="Relative Yaw")
    )
    fig.add_trace(go.Scatter(
        x=subjects[subject].index,
        y=subjects[subject]["airpodsYawDegrees"],
        mode="lines",
        name="Raw Yaw Airpods")
    )
    fig.add_trace(go.Scatter(
        x=subjects[subject].index,
        y=subjects[subject]["phoneYawDegrees"],
        mode="lines",
        name="Raw Yaw iPhone")
    )
    fig.add_annotation(
        x=timestamp,
        y=subjects[subject]["relativeYaw"].max(),
        text="Chosen Time " + str(timestamp),
        showarrow=False,
        yshift=10
    )
    fig.update_layout(
        title_text="Yaws",
        autosize=True,
        hovermode='closest',
        shapes = [
            # Line Horizontal
            {
                'type': 'line',
                'x0': timestamp,
                'y0': subjects[subject]["relativeYaw"].min(),
                'x1': timestamp,
                'y1': subjects[subject]["relativeYaw"].max(),
                'line': {
                    'color': 'rgb(50, 171, 96)',
                    'width': 2
                },
            }
        ]
    )
    add_conditions(subject,fig,"airpodsYawDegrees")
    fig.update_xaxes(title_text='Time(seconds)')
    fig.update_yaxes(title_text='Yaw (Degrees)')

    return fig
#Rolls
@dash.callback(
    dash.dependencies.Output('rolls-scatter', 'figure'),
    [dash.dependencies.Input('filter_subject', 'value'),
     dash.dependencies.Input('filter-slider-timestamp', 'value')])
def update_rolls(subject, timestamp):
    if not subject in subjects:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=subjects[subject].index,
        y=subjects[subject]["relativeRoll"],
        mode="lines",
        name="Relative Roll")
    )
    fig.add_trace(go.Scatter(
        x=subjects[subject].index,
        y=subjects[subject]["airpodsRollDegrees"],
        mode="lines",
        name="Raw Roll Airpods")
    )
    fig.add_trace(go.Scatter(
        x=subjects[subject].index,
        y=subjects[subject]["phoneRollDegrees"],
        mode="lines",
        name="Raw Roll iPhone")
    )
    fig.add_annotation(
        x=timestamp,
        y=subjects[subject]["relativeRoll"].max(),
        text="Chosen Time " + str(timestamp),
        showarrow=False,
        yshift=10
    )
    fig.update_layout(
        title_text="Rolls (Body/Bike Tilt Angle ?)",
        autosize=True,
        hovermode='closest',
        shapes = [
            # Line Horizontal
            {
                'type': 'line',
                'x0': timestamp,
                'y0': subjects[subject]["relativeRoll"].min(),
                'x1': timestamp,
                'y1': subjects[subject]["relativeRoll"].max(),
                'line': {
                    'color': 'rgb(50, 171, 96)',
                    'width': 2
                },
            }
        ]
    )
    add_conditions(subject,fig,"airpodsRollDegrees")
    fig.update_xaxes(title_text='Time(seconds)')
    fig.update_yaxes(title_text='Roll (Degrees)')

    return fig
#Pitchs
@dash.callback(
    dash.dependencies.Output('pitchs-scatter', 'figure'),
    [dash.dependencies.Input('filter_subject', 'value'),
     dash.dependencies.Input('filter-slider-timestamp', 'value')])
def update_pitchs(subject, timestamp):
    if not subject in subjects:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=subjects[subject].index,
        y=subjects[subject]["relativePitch"],
        mode="lines",
        name="Relative Pitch")
    )
    fig.add_trace(go.Scatter(
        x=subjects[subject].index,
        y=subjects[subject]["airpodsPitchDegrees"],
        mode="lines",
        name="Raw Pitch Airpods")
    )
    fig.add_trace(go.Scatter(
        x=subjects[subject].index,
        y=subjects[subject]["phonePitchDegrees"],
        mode="lines",
        name="Raw Pitch iPhone")
    )
    fig.add_annotation(
        x=timestamp,
        y=subjects[subject]["relativePitch"].max(),
        text="Chosen Time " + str(timestamp),
        showarrow=False,
        yshift=10
    )
    fig.update_layout(
        title_text="Pitchs (Bike Altitude / Look at Phone ?)",
        autosize=True,
        hovermode='closest',
        shapes = [
            # Line Horizontal
            {
                'type': 'line',
                'x0': timestamp,
                'y0': subjects[subject]["relativePitch"].min(),
                'x1': timestamp,
                'y1': subjects[subject]["relativePitch"].max(),
                'line': {
                    'color': 'rgb(50, 171, 96)',
                    'width': 2
                },
            }
        ]
    )
    add_conditions(subject,fig,"airpodsPitchDegrees")
    fig.update_xaxes(title_text='Time(seconds)')
    fig.update_yaxes(title_text='Pitch (Degrees)')

    return fig
# Altitude
@dash.callback(
    dash.dependencies.Output('altitude', 'figure'),
    [dash.dependencies.Input('filter_subject', 'value'),
     dash.dependencies.Input('filter-slider-timestamp', 'value')])
def update_altitude(subject, timestamp):
    if not subject in subjects:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=subjects[subject].index,
        y=subjects[subject]["locationData.altitude"],
        mode="lines",fill="tozeroy")
    )
    fig.add_annotation(
        x=timestamp,
        y=subjects[subject]["locationData.altitude"].max(),
        text="Chosen Time " + str(timestamp),
        showarrow=False,
        yshift=10
    )
    fig.update_layout(
        title_text="Altitude",
        autosize=True,
        hovermode='closest',
        shapes = [
        # Line Horizontal
        {
            'type': 'line',
            'x0': timestamp,
            'y0': subjects[subject]["locationData.altitude"].min(),
            'x1': timestamp,
            'y1': subjects[subject]["locationData.altitude"].max(),
            'line': {
                'color': 'rgb(50, 171, 96)',
                'width': 2
            },
        }]
    )
    add_conditions(subject,fig,"locationData.altitude")
    fig.update_xaxes(title_text='Time (seconds)')
    fig.update_yaxes(
        title_text='Altitude (meters)',
        range = [subjects[subject]["locationData.altitude"].min(),subjects[subject]["locationData.altitude"].max()]
    )

    return fig

#Acceleration
@dash.callback(
    dash.dependencies.Output('acceleration', 'figure'),
    [dash.dependencies.Input('filter_subject', 'value'),
     dash.dependencies.Input('filter-slider-timestamp', 'value')])
def update_acceleration(subject, timestamp):
    if not subject in subjects:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=subjects[subject].index,
        y=subjects[subject]["phoneAcceleration.y"],
        mode="lines")
    )
    fig.add_annotation(
        x=timestamp,
        y=subjects[subject]["phoneAcceleration.y"].max(),
        text="Chosen Time " + str(timestamp),
        showarrow=False,
        yshift=10
    )
    fig.update_layout(
        title_text="Acceleration",
        autosize=True,
        hovermode='closest',
        shapes = [
        # Line Horizontal
        {
            'type': 'line',
            'x0': timestamp,
            'y0': subjects[subject]["phoneAcceleration.y"].min(),
            'x1': timestamp,
            'y1': subjects[subject]["phoneAcceleration.y"].max(),
            'line': {
                'color': 'rgb(50, 171, 96)',
                'width': 2
            },
        }]
    )
    add_conditions(subject,fig,"phoneAcceleration.y")
    fig.update_xaxes(title_text='Time (seconds)')
    fig.update_yaxes(title_text='Acceleration Y')
    return fig

#Velocity
@dash.callback(
    dash.dependencies.Output('velocity', 'figure'),
    [dash.dependencies.Input('filter_subject', 'value'),
     dash.dependencies.Input('filter-slider-timestamp', 'value')])
def update_velocity(subject, timestamp):
    if not subject in subjects:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=subjects[subject].index,
        y=subjects[subject]["locationData.velocity"],
        mode="lines")
    )
    fig.add_annotation(
        x=timestamp,
        y=subjects[subject]["locationData.velocity"].max(),
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
            'y0': subjects[subject]["locationData.velocity"].min(),
            'x1': timestamp,
            'y1': subjects[subject]["locationData.velocity"].max(),
            'line': {
                'color': 'rgb(50, 171, 96)',
                'width': 2
            },
        }]
    )
    add_conditions(subject,fig,"locationData.velocity")
    fig.update_xaxes(title_text='Time (seconds)')
    fig.update_yaxes(title_text='Velocity (km/h)')

    return fig

#Headline
@dash.callback(
    dash.dependencies.Output('headline_subject', 'children'),
    [dash.dependencies.Input('filter_subject', 'value')])
def update_headline(subject):
    return "### " + subject

#Phone Battery
@dash.callback(
    dash.dependencies.Output('phone_battery', 'figure'),
    [dash.dependencies.Input('filter_subject', 'value'),
     dash.dependencies.Input('filter-slider-timestamp', 'value')])
def update_battery(subject, timestamp):
    if not subject in subjects:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=subjects[subject].index,
        y=subjects[subject]["phoneBattery"],
        fill="tozeroy")
    )
    fig.add_annotation(
        x=timestamp,
        y=subjects[subject]["phoneBattery"].max(),
        text="Chosen Time " + str(timestamp),
        showarrow=False,
        yshift=10
    )
    fig.update_layout(
        title_text="Phone Battery",
        autosize=True,
        hovermode='y unified',
        shapes = [
        # Line Horizontal
        {
            'type': 'line',
            'x0': timestamp,
            'y0': subjects[subject]["phoneBattery"].min(),
            'x1': timestamp,
            'y1': subjects[subject]["phoneBattery"].max(),
            'line': {
                'color': 'rgb(50, 171, 96)',
                'width': 2
            },
        }]
    )
    fig.update_xaxes(title_text='Time (seconds)')
    fig.update_yaxes(title_text='Battery level %')

    return fig
#-------------------#
#------Server-------#
#-------------------#

if __name__ == '__main__':
    app.run_server(debug=True)

#app = JupyterDash(__name__, external_stylesheets=external_stylesheets)
#app.run_server(mode='external', port = 8090, dev_tools_ui=True, debug=True,dev_tools_hot_reload =True, threaded=True)