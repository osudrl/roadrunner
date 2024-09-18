import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import dash_bootstrap_components as dbc
import dash
import pickle
# import plotly
# from chart_studio.plotly import plot, iplot
from copy import deepcopy
from plotly.subplots import make_subplots
from pathlib import Path

from dash import Dash, dcc, html, Input, Output, State, callback, ALL, ctx


SIDEBAR_SIZE = 150
DASH_STYLES = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]
DASH_INDEX = 0

app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    # Page Title
    html.H4('Hardware Log Viewer', style={'margin-left': str(SIDEBAR_SIZE+40)+"px"}),
    # Text input for path to logging directory
    html.Div(id='input-path-div',
             children=[dcc.Input(
                 id='input-path',
                 type='text',
                 value="./",
                 placeholder='Path to data directory',
                 style={'width': '400px'}
                ),
                dbc.Button("Load", id='submit', n_clicks=0)],
             style={'margin-left': str(SIDEBAR_SIZE+40)+"px", 'margin-bottom': '10px'}),
    # Buttons for path suggestions
    html.Div(id='path-suggest', children=[], style={'margin-left': str(SIDEBAR_SIZE+40)+"px", 'margin-bottom': '10px'}),
    # html.Div(children=[
    #     # Path name suggestions
    #     dbc.DropdownMenu(
    #         id='path-suggest',
    #         label=dcc.Input(
    #             id='input-path',
    #             type='text',
    #             value="./",
    #             placeholder='Path to data directory',
    #             style={'width': '400px'}
    #         ),
    #         children=[],
    #         style={"overflow-y": "scroll"}
    #     ),
    #     dbc.Button("Load", id='submit', n_clicks=0, style={'display': 'inline-block', 'height': '25px', 'margin-left': '10px'})
    # ], style={'margin-left': str(SIDEBAR_SIZE+40)+"px", 'margin-bottom': '80px'}),
    # Sidebar to display log data dictionary keys
    html.Div(id="sidebar",
            style={"position": "fixed", "top": 0, "left": 0, "bottom": 0,
                   "width": str(SIDEBAR_SIZE)+"px", "padding": "20px",
                   "background-color": "#f8f9fa", "overflow": "scroll"},
            children=[]),
    # Checklist to select which logs to plot
    html.Div(id='load-paths',
             children=[dcc.Checklist(id="load-paths-checklist", options=[], value=[], labelStyle={"display": "block"}, style={"overflow": "auto"})],
             style={'display': 'block', "overflow": "scroll", 'margin-left': str(SIDEBAR_SIZE+40)+"px", 'margin-bottom': '10px'}),
    # Actual graph plots
    html.Div(id="plots", children=[], style={'margin-left': str(SIDEBAR_SIZE+40)+"px"}),
    # Shared store object of all loaded log data
    dcc.Store(id='all-data', data={}),
    dcc.Store(id='path-dash-style', data={}),
    dcc.Store(id='dash-style-index', data=0),
    dcc.Store(id='input-color', data={}),
    dcc.Store(id='output-color', data={})
])


# Callback to update suggestions as dropdown selections
# Based off this example https://community.plotly.com/t/dcc-input-with-suggestions-based-on-text-similarity/73420/10
@callback(
    Output('path-suggest', 'children'),
    Input('input-path', 'value'),
    prevent_initial_call=True
)
def update_suggestions(value):
    full_path = os.path.expanduser(value)
    if not os.path.isdir(full_path):
        raise dash.exceptions.PreventUpdate
    suggest_dir = []
    # Only include non-hidden files and non-directories
    for f in os.listdir(full_path):
        if (not f.startswith(".")) and (os.path.isdir(os.path.join(full_path, f))):
            suggest_dir.append(f)
    return [dbc.Button(path, id={'type': 'selection', 'index': path}, n_clicks=0, style={'margin': "2px"}) for path in suggest_dir]
    # return [
    #     dbc.DropdownMenuItem(
    #         id={'type': 'selection', 'index': idx},
    #         children=match,
    #         # style={'display': 'block', "overflow": "auto"}
    #         # style={'display': 'flex'}
    #         # style={"overflow-y": "scroll"}
    #     ) for idx, match in enumerate(suggest_dir)
    # ]

# Callback to update the text input if user selects option from the dropdown
@callback(
    Output('input-path', 'value'),
    Input('input-path', 'value'),
    Input({'type': 'selection', 'index': ALL}, 'n_clicks'),
    State({'type': 'selection', 'index': ALL}, 'children'),
    prevent_initial_callback=True,
    prevent_initial_call=True
)
def update_input(input_path, clicks, select):
    # If nothing has been clicked
    if all(click == 0 for click in clicks):
        raise dash.exceptions.PreventUpdate

    # Get trigger index
    idx = ctx.triggered_id['index']
    if not idx:
        raise dash.exceptions.PreventUpdate

    full_path = os.path.join(input_path, idx)
    return full_path

# Callback to load data when user clicks submit button
@callback(
    Output('all-data', 'data'),
    Output('sidebar', 'children'),
    Output('plots', 'children'),
    Output('load-paths-checklist', 'options'),
    Output('path-dash-style', 'data'),
    Output('dash-style-index', 'data'),
    Output('input-color', 'data'),
    Output('output-color', 'data'),
    Input('submit', 'n_clicks'),
    State('all-data', 'data'),
    State("plots", "children"),
    State('input-path', 'value'),
    State('load-paths-checklist', 'options'),
    State('sidebar', 'children'),
    State('path-dash-style', 'data'),
    State('dash-style-index', 'data'),
    State('input-color', 'data'),
    State('output-color', 'data'),
    prevent_initial_call=True
)
def load_data(n_clicks, data, plots, value, load_path_checklist, sidebar, path_dash_style, dash_style_index, input_colors, output_colors):
    full_path = os.path.expanduser(value)
    if not os.path.isdir(full_path):
        raise dash.exceptions.PreventUpdate
    pickle_data = {}
    # Loop through directory and find all pickle files
    for filename in os.listdir(full_path):
        if ".pkl" in filename:
            name_split = filename.split("part")
            part_num = int(name_split[1][0])
            pickle_data[part_num] = pickle.load(open(os.path.join(full_path, filename), "rb"))
    # Combine all log files into single dictionary
    curr_data = {}
    if len(pickle_data.keys()) > 0:
        curr_data = pickle_data[0]
    for i in range(1, len(pickle_data.keys())):
        for key, val in pickle_data[i].items():
            if isinstance(val, dict):    # If is a dict, need to handle combining
                for key2, val2 in val.items():
                    curr_data[key][key2] += val2
            else:   # else is a list and can just add
                curr_data[key] += val
    curr_data["time"] = np.array(curr_data["time"]) - curr_data["time"][0]
    time = curr_data["time"]
    path_dash_style[value] = DASH_STYLES[dash_style_index]
    dash_style_index += 1

    # Add dictionary to all data
    data[value] = curr_data

    # Add to checklist of loaded paths
    load_path_checklist.append(value)

    # Create sidebar and graphs only if this is the first time loading data
    if plots != []:
        return data, sidebar, plots, load_path_checklist, path_dash_style, dash_style_index, input_colors, output_colors
    else:
        sidebar = []
        graphs = []
        for key, val in curr_data.items():
            if plots == []:
                graphs.append(dcc.Graph(id={"type": "plot", "index": key},
                    figure={'layout': {'title': f"{key}", "margin": {"l": 40, "t": 40}}}, style={"display": "none"}))
            # Add toggle button for this category
            sidebar.append(dbc.Button(key, id={"type": "plot-type", "index": n_clicks}, className="mb-3", color="primary", n_clicks=0))
            # Form checklists
            checklist = []
            options = []
            if isinstance(val, dict):
                for key2, val2 in val.items():
                    options.append(key + "/" + key2)
            else:
                options.append(key + "/" + key)
            checklist = dcc.Checklist(id={"type":"plot-checklist", "index": key}, options=options, value=[], style={"overflow": "auto"}, labelStyle={"display": "block"})
            # sidebar.append(dbc.Collapse(checklist, id={"type":"plot-type-collapse", "index": n_clicks}, is_open=False))
            sidebar.append(checklist)

        # Make color dicts for input and output plots
        output_colors = {}
        input_colors = {}
        color_ind = 0
        for key, val in curr_data["output"].items():
            output_colors[key] = px.colors.qualitative.Alphabet[color_ind]
            color_ind += 1
            if color_ind >= len(px.colors.qualitative.Alphabet):
                color_ind = 0
        color_ind = 0
        for key, val in curr_data["input"].items():
            input_colors[key] = px.colors.qualitative.Alphabet[color_ind]
            color_ind += 1
            if color_ind >= len(px.colors.qualitative.Alphabet):
                color_ind = 0

        return data, sidebar, graphs, load_path_checklist, path_dash_style, dash_style_index, input_colors, output_colors

# @callback(
#     # Output({'type': 'plot-type-collapse', 'index': dash.MATCH}, 'is_open'),
#     Input({'type': 'plot-type', 'index': dash.MATCH}, 'n_clicks'),
#     # State({'type': 'plot-type-collapse', 'index': dash.MATCH}, 'is_open'),
#     State({'type': 'plot-type-collapse', 'index': dash.MATCH}, 'id')
# )
# def toggle_collapse(n_clicks, is_open, id):
#     print("nclicks", n_clicks, "is_open", is_open, "id", id)
#     if n:
#         print("return not open")
#         return not is_open
#     return is_open

@callback(
    Output({"type": "plot", "index": dash.MATCH}, "style"),
    Output({"type": "plot", "index": dash.MATCH}, "figure"),
    Input({'type': 'plot-checklist', 'index': dash.MATCH}, 'value'),
    Input('load-paths-checklist', 'value'),
    State({'type': 'plot', 'index': dash.MATCH}, 'figure'),
    State('all-data', 'data'),
    State('path-dash-style', 'data'),
    State('input-color', 'data'),
    State('output-color', 'data')
)
def update_graph_show(data_type_checklist, path_name, fig, all_data, path_dash_style, input_colors, output_colors):
    # Based on the current selected load-paths and the plot-checklist, form dictionary of all data
    # that needs to be plotted. Dict will be {path-name: data-type}
    to_plot = {path: deepcopy(data_type_checklist) for path in path_name}

    if "data" in fig.keys():
        # Remove plots that were unchecked
        temp_data = deepcopy(fig["data"])
        for data in temp_data:
            if data["meta"][0] != "legend":
                if data["meta"][0] not in to_plot.keys():
                    fig["data"].remove(data)
                elif data["meta"][1] + "/" + data["name"] not in to_plot[data["meta"][0]]:
                    fig["data"].remove(data)
        # Disinclude plots that have already been made
        for data in fig["data"]:
            if data["meta"][0] in to_plot.keys() and data["meta"][1] + "/" + data["name"] in to_plot[data["meta"][0]]:
                to_plot[data["meta"][0]].remove(data["meta"][1] + "/" + data["name"])
    plots = []
    for key, val in to_plot.items():
        time = all_data[key]["time"]
        for data_type in val:
            split = data_type.split("/")
            data_type = split[0]
            joint = split[1]
            if isinstance(all_data[key][data_type], dict):
                data = all_data[key][data_type][joint]
            else:
                data = all_data[key][data_type]
            # print("using dash style", path_dash_style[key])
            if data_type == "input":
                color = input_colors[joint]
            elif data_type == "output":
                color = output_colors[joint]
            else:
                color = "#3283FE"
            plots.append({'x': time, 'y': data, "name":joint, "line": {"dash": path_dash_style[key], "color": color}, "showlegend":False, "meta":[key, data_type]})

    # Add/remove dummy plots for legend. There's probably a better way to do this without the
    # multiple loops
    if "data" in fig.keys():
        legend_names = set()
        path_legend_names = set()
        for data in data_type_checklist:
            legend_names.add(data.split("/")[1])
        if data_type_checklist:
            for data in path_name:
                path_legend_names.add(data)

        # Loop through all current plots to find already existing legends (don't need to plot, can
        # remove from legend_names) and unneeded legends (remove from fig["data"])
        for plot in temp_data:
            if plot["meta"][0] == "legend":
                if plot["name"] not in legend_names and plot["name"] not in legend_names:    # Legend plot not needed, remove
                    fig["data"].remove(plot)
                elif plot["name"] in legend_names: # Legend plot already exists, remove from legend_names
                    legend_names.remove(plot["name"])
                elif plot["name"] in path_legend_names:
                    path_legend_names.remove(plot["name"])
                # if plot["name"] not in path_legend_names:    # Legend plot not needed, remove
                    # fig["data"].remove(plot)
                # else: # Legend plot already exists, remove from legend_names
                    # path_legend_names.remove(plot["name"])
        # Create the new legends
        for name in legend_names:
            if data_type_checklist:
                type_split = data_type_checklist[0].split("/")
                if type_split[0] == "input":
                    color = input_colors[name]
                elif type_split[0] == "output":
                    color = output_colors[name]
                else:
                    color = "#3283FE"
                plots.append({'x': [None], 'y': [None], "name": name, "line": {"dash": "solid", "color": color}, "showlegend":True, "meta":["legend", name]})
        for name in path_legend_names:
            # newline = np.arange(1, len(name) // 20)
            newline = len(name) // 20
            name_wrap = ""
            for i in range(newline):
                name_wrap += name[20*i:20*(i+1)] + "<br>"
            name_wrap = name_wrap[:-4]
            plots.append({'x': [None], 'y': [None], "name": name_wrap, "line": {"dash": path_dash_style[name], "color": "black"}, "showlegend":True, "meta":["legend", name]})


    if not "data" in fig.keys():
        fig["data"] = plots
    else:
        fig["data"] += plots

    if fig["data"] == []:
        style = {"display": "none"}
    else:
        style = {"display": "block"}
    return style, fig


if __name__ == "__main__":
    app.run_server(debug=True)