
# coding: utf-8

# In[138]:

import pandas as pd
import numpy as np
import xarray as xr
import datashader as ds
import datashader.transfer_functions as tf
from collections import OrderedDict
import plotly.plotly as py
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import dash
import json
import copy


# In[2]:

n = 1000000

np.random.seed(2)
cols = ['Signal']                    # Column name of signal
start = 1456297053                   # Start time
end = start + n

time = np.linspace(start, end, n)


# In[139]:

cols = ['Signal']                    # Column name of signal
start = 1456297053                   # Start time
end = start + n                      # End time

# Generate a fake signal
time = np.linspace(start, end, n)
signal = np.random.normal(0, 0.3, size=n).cumsum() + 50

# Generate many noisy samples from the signal
noise = lambda var, bias, n: np.random.normal(bias, var, n)
data = {c: signal + noise(1, 10*(np.random.random() - 0.5), n) for c in cols}


# # Pick a few samples and really blow them out
locs = np.random.choice(n, 10)
# print locs
data['Signal'][locs] *= 2

# # Default plot ranges:
x_range = (start, end)
y_range = (1.2 * signal.min(), 1.2 * signal.max())


# Create a dataframe
data['Time'] = np.linspace(start, end, n)
df = pd.DataFrame(data)

time_start = df['Time'].values[0]
time_end = df['Time'].values[-1]


# In[5]:

cvs = ds.Canvas(x_range = x_range, y_range=y_range)

aggs = OrderedDict((c, cvs.line(df, 'Time', c)) for c in cols)
img = tf.shade(aggs['Signal'])


# In[6]:

arr = np.array(img)
z = arr.tolist()

# axes
dims = len(z[0]), len(z)


# In[175]:

app = dash.Dash()
server = app.server
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})


# In[169]:

fig = {'data':
        [{'x': np.linspace(x_range[0], x_range[1], dims[0]),
          'y': np.linspace(y_range[0], y_range[1], dims[0]),
          'z': z,
          'type': 'heatmap',
          'showscale': False,
          'colorscale': [[0, 'rgba(255, 255, 255,0)'], [1, 'rgba(0,0,255,1)']]}],
        'layout': {'margin': {'t': 50, 'b': 0}, 'yaxis': {'fixedrange': True}}
      }


# In[170]:

app.layout = html.Div([
        html.Div([
                html.Div([
                        html.H2('Visualize Data'),
                        html.H5('Decimated view of time series.', id = 'header-1'),
                        dcc.Graph(id = 'graph-1', figure = fig)], className = 'twelve columns')], className ='row'),
        html.Div([
                html.Div([
                        html.H5('Range Selection: ', id = 'header-2'),
                        html.H6(id = 'header-2-b'),
                        dcc.Graph(id = 'graph-2', figure = fig)], className = 'twelve columns')], className ='row'),
        html.Div([
                html.Div([
                        html.H5('High-res view of selected data. Select a view containing less than 10000 points.', id = 'header-3'),
                        dcc.Graph(id = 'graph-3', figure = [{'x': [], 'y': [], 'type': 'pointcloud'}])], className = 'twelve columns')], className ='row'),
    ])


# In[171]:

@app.callback(
    Output('header-2-b', 'children'),
    [Input('graph-1', 'relayoutData')])
def selectionRange(selection):
    if selection is not None:
        return "Selecting {0} to {1}: {2} points.".format(selection['xaxis.range[0]'], selection['xaxis.range[1]'],  int(selection['xaxis.range[1]'] - selection['xaxis.range[0]']))
    else:
        return "Selecting 0 points."

@app.callback(
    Output('graph-2', 'figure'),
    [Input('graph-1', 'relayoutData')])
def selectionHighlight(selection):
    if 'xaxis.range[0]' in selection and 'xaxis.range[1]' in selection:
        shape = dict(
            type = 'rect',
            xref = 'x',
            yref = 'paper',
            y0 = 0,
            y1 = 1,
            x0 = selection['xaxis.range[0]'],
            x1 = selection['xaxis.range[1]'],
            line = {
                'color': 'rgba(255, 0, 0, 1)',
                'width': 1,
            },
            fillcolor = 'rgba(255, 0, 0, 0.7)',
        )
        new_figure = fig.copy()
        new_figure['layout']['shapes'] = [shape]
    return new_figure


# In[172]:

@app.callback(
    Output('graph-3', 'figure'),
    [Input('graph-1', 'relayoutData')])
def draw_undecimated_data(selection):
    pt_cloud = [dict(x=[], y=[], type='pointcloud')]
    if 'xaxis.range[0]' in selection and 'xaxis.range[1]' in selection:
        x0 = selection['xaxis.range[0]']
        x1 = selection['xaxis.range[1]']
        # Slice the dataframe
        sub_df = df[(df.Time >= x0) & (df.Time <= x1)]
        num_pts = len(sub_df)
        if num_pts < 10000:
            pt_cloud_data = [dict(x=sub_df['Time'], y=sub_df['Signal'],
                             type='scattergl', marker=dict(sizemin=1, sizemax=30, color='darkblue'))]
            pt_cloud_layout = dict( margin = dict( t=0, b=0 ) )
            pt_cloud = dict( data = pt_cloud_data, layout = pt_cloud_layout )
    return pt_cloud


# In[173]:

if __name__ == '__main__':
    app.run_server(port = 7000)


# In[ ]:




# In[ ]:




# In[ ]:

# app.layout = html.Div(children = [
#         dcc.Graph(id = 'graph',
#                   figure = {'data':
#                             [{'x': np.linspace(x_range[0], x_range[1], dims[0]),
#                               'y': np.linspace(y_range[0], y_range[1], dims[0]),
#                               'z': z,
#                               'type': 'heatmap',
#                               'showscale': False,
#                               'colorscale': [[0, 'rgba(255,255,255,0)'], [1, 'rgba(0,0,255,1)']]}],

#                             'layout': {'title': 'Starter', 'zoom': 'zoom', 'margin': {'t':50, 'b':0}}
#                            })
#     ])
