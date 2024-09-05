import argparse
import dash
import dash_bootstrap_components as dbc
import logging
import numpy as np
import os
import pickle
import plotly.express as px
import plotly.graph_objects as go

from copy import deepcopy
from colors import FAIL, ENDC
from dash import Dash, dcc, html, Input, Output, State, callback, ALL, ctx

SIDEBAR_SIZE = 200
DASH_STYLES = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]
DASH_INDEX = 0
LLAPI_SUBSAMPLE = 10

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Create app and define layout
app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
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
                dbc.Button("Load", id='submit', n_clicks=0, style={'margin-left': '10px'})],
             style={'margin-left': str(SIDEBAR_SIZE+40)+"px", 'margin-bottom': '10px'}),
    # Buttons for path suggestions
    html.Div(id='path-suggest',
             children=[],
             style={'margin-left': str(SIDEBAR_SIZE+40)+"px", 'margin-bottom': '10px'}),
    # Sidebar to display log data dictionary keys
    html.Div(id="sidebar",
            style={"position": "fixed", "top": 0, "left": 0, "bottom": 0,
                   "width": str(SIDEBAR_SIZE)+"px", "padding": "20px",
                   "background-color": "#f8f9fa", "overflow": "scroll", "display": "grid"},
            children=[]),
    # Checklist to select which logs to plot
    html.Div(id='load-paths',
             children=[dcc.Checklist(id="load-paths-checklist",
                options=[],
                value=[],
                labelStyle={"display": "block"},
                style={"overflow": "auto"})],
             style={'display': 'block',
                    'overflow': 'scroll',
                    'margin-left': str(SIDEBAR_SIZE+40)+"px",
                    'margin-bottom': '10px'}),
    # Actual graph plots
    html.Div(id="plots", children=[], style={'margin-left': str(SIDEBAR_SIZE+40)+"px"}),
    # Shared store object of all loaded log data
    dcc.Store(id='all-data', data={}),
    # Dictionary to store the dash style of each path
    dcc.Store(id='path-dash-style', data={}),
    # Store to keep track of the current dash style index
    dcc.Store(id='dash-style-index', data=0),
    # Dictionary of input joint to plot color
    dcc.Store(id='input-color', data={}),
    # Dictionary of output joint to plot color
    dcc.Store(id='output-color', data={}),
    # Dictionary of llapi data to plot color
    dcc.Store(id='llapi-color', data={}),
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
    suggest_dir = sorted(suggest_dir)
    return [dbc.Button(path, id={'type': 'selection', 'index': path},
            n_clicks=0, style={'margin': "2px"}) for path in suggest_dir]

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
    Output('llapi-color', 'data'),
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
    State('llapi-color', 'data'),
    prevent_initial_call=True
)
def load_data(n_clicks, data, plots, value, load_path_checklist, sidebar, path_dash_style,
              dash_style_index, input_colors, output_colors, llapi_colors):
    full_path = os.path.expanduser(value)
    if not os.path.isdir(full_path):
        raise dash.exceptions.PreventUpdate
    pickle_data = {}
    # Loop through directory and find all pickle files
    find_pkl = False
    for filename in os.listdir(full_path):
        if ".pkl" in filename:
            name_split = filename.split("part")
            part_num = int(name_split[1][0])
            pickle_data[part_num] = pickle.load(open(os.path.join(full_path, filename), "rb"))
            find_pkl = True
    if not find_pkl:
        print(f"{FAIL}Error: No pickle files found in directory, invalid log file path{ENDC}")
        raise dash.exceptions.PreventUpdate
    # Combine all log files into single dictionary
    curr_data = {}
    if len(pickle_data.keys()) > 0:
        curr_data = pickle_data[0]
    for i in range(1, len(pickle_data.keys())):
        for key, val in pickle_data[i].items():
            if isinstance(val, dict):    # If is a dict, need to handle combining
                for key2, val2 in val.items():
                    if isinstance(val2, dict):
                        for key3, val3 in val2.items():
                            if key == "llapi":
                                curr_data[key][key2][key3] += val3[1::LLAPI_SUBSAMPLE]
                            else:
                                curr_data[key][key2][key3] += val3[1:]
                    else:
                        if key == "llapi":
                            curr_data[key][key2] += val2[1::LLAPI_SUBSAMPLE]
                        else:
                            curr_data[key][key2] += val2[1:]
            elif key == "flags":
                for flags in val:
                    flags[0] += len(pickle_data[i]["time"])
                    flags[1] += len(pickle_data[i]["llapi"]["time"])
                curr_data[key] += val
            else:   # else is a list and can just add
                if key == "llapi":
                    curr_data[key] += val[1::LLAPI_SUBSAMPLE]
                else:
                    curr_data[key] += val[1:]

    for key, val in curr_data.items():
        if isinstance(val, dict):
            for key2, val2 in val.items():
                if isinstance(val2, dict):
                    for key3, val3 in val2.items():
                        if key == "llapi":
                            curr_data[key][key2][key3] = val3[1::LLAPI_SUBSAMPLE]
                        else:
                            curr_data[key][key2][key3] = val3[1:]
                else:
                    if key == "llapi":
                        curr_data[key][key2] = val2[1::LLAPI_SUBSAMPLE]
                    else:
                        curr_data[key][key2] = val2[1:]
        elif key != "flags":
            curr_data[key] = val[1:]
    curr_data["time"] = np.array(curr_data["time"]) - curr_data["time"][0]
    curr_data["llapi"]["time"] = np.array(curr_data["llapi"]["time"]) - curr_data["llapi"]["time"][0]
    for flag in curr_data["flags"]:
        flag[1] = int(flag[1] / LLAPI_SUBSAMPLE)
    path_dash_style[value] = DASH_STYLES[dash_style_index]
    dash_style_index += 1

    # Add dictionary to all data
    data[value] = curr_data

    # Add to checklist of loaded paths
    load_path_checklist.append(value)

    # Create sidebar and graphs only if this is the first time loading data
    if plots != []:
        return data, sidebar, plots, load_path_checklist, path_dash_style, dash_style_index, \
               input_colors, output_colors, llapi_colors
    else:
        sidebar = []
        graphs = []
        for key, val in curr_data.items():
            if key == "flags":
                sidebar.append(dbc.Button(key,
                                      id={"type": "plot-type", "index": key},
                                      className="mb-3",
                                      color="primary",
                                      n_clicks=0))
                sidebar.append(dbc.Collapse(
                                    [dcc.Checklist(id="flags",
                                    options=["flags"],
                                    value=[],
                                    style={"overflow": "auto"},
                                    labelStyle={"display": "block"})],
                                id={"type": "collapse", "index": "flags"}, is_open=False))
                continue
            if plots == []:
                graphs.append(dcc.Graph(id={"type": "plot", "index": key},
                    figure={'layout': {'title': f"{key}", "margin": {"l": 40, "t": 40}}},
                    style={"display": "none"}))
            # Add toggle button for this category
            sidebar.append(dbc.Button(key,
                                      id={"type": "plot-type", "index": key},
                                      className="mb-3",
                                      color="primary",
                                      n_clicks=0))
            # Form checklists
            checklist = []
            options = []
            if isinstance(val, dict):
                for key2, val2 in val.items():
                    if isinstance(val2, dict):
                        graphs.append(dcc.Graph(id={"type": "plot", "index": key+"/"+key2},
                            figure={'layout': {'title': f"{key}/{key2}", "margin": {"l": 40, "t": 40}}},
                            style={"display": "none"}))
                        checklist.append(dbc.Button(key2,
                                                  id={"type": "plot-type", "index": key2},
                                                  className="mb-3",
                                                  color="primary",
                                                  n_clicks=0,
                                                  style={'margin-left': "20px", "display": "grid"}))
                        options_lower = []
                        for key3, val3 in val2.items():
                            options_lower.append(key3)

                        lower_checklist = dcc.Checklist(id={"type":"plot-checklist", "index": key+"/"+key2},
                                                     options=options_lower,
                                                     value=[],
                                                     style={"overflow": "auto", 'margin-left': "20px"},
                                                     labelStyle={"display": "block"})
                        lower_collapse = dbc.Collapse(lower_checklist, id={"type": "collapse", "index": key2}, is_open=False)
                        checklist.append(lower_collapse)
                    else:
                        options.append(key + "/" + key2)
            else:
                options.append(key + "/" + key)
            checklist.append(dcc.Checklist(id={"type":"plot-checklist", "index": key},
                                    options=options,
                                    value=[],
                                    style={"overflow": "auto"},
                                    labelStyle={"display": "block"}))
            collapse = dbc.Collapse(checklist, id={"type": "collapse", "index": key}, is_open=False)
            sidebar.append(collapse)

        # Make color dicts for input and output plots
        output_colors = {}
        input_colors = {}
        llapi_colors = {}
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
        color_ind = 0
        if "llapi" in curr_data.keys():
            for key, val in curr_data["llapi"].items():
                if isinstance(val, dict):
                    llapi_colors[key] = {}
                    color_ind = 0
                    for key2, val2 in val.items():
                        llapi_colors[key][key2] = px.colors.qualitative.Alphabet[color_ind]
                        color_ind += 1
                        if color_ind >= len(px.colors.qualitative.Alphabet):
                            color_ind = 0
                else:
                    llapi_colors[key] = px.colors.qualitative.Alphabet[0]

        return data, sidebar, graphs, load_path_checklist, path_dash_style, dash_style_index, \
               input_colors, output_colors, llapi_colors

@callback(
    Output({"type": "collapse", "index": dash.MATCH}, "is_open"),
    [Input({"type": "plot-type", "index": dash.MATCH}, "n_clicks")],
    [State({"type": "collapse", "index": dash.MATCH}, "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

@callback(
    Output('plots', 'children', allow_duplicate=True),
    Input("flags", "value"),
    State('plots', 'children'),
    State('all-data', 'data'),
    State('path-dash-style', 'data'),
    prevent_initial_call=True
)
def update_flag_check(flag, plots, all_data, path_dash_style):
    if flag:
        for fig in plots:
            if "data" in fig["props"]["figure"].keys():
                skip_plot = False
                path_set = set()
                for plot in fig["props"]["figure"]["data"]:
                    if "name" in plot.keys() and plot["name"] == "flag":
                        skip_plot = True
                    elif "meta" in plot.keys() and plot["meta"][0] != "legend":
                        path_set.add(plot["meta"][0])
                        plot_type = plot["meta"][1]
                if not skip_plot:
                    y_range = fig["props"]["figure"]["layout"]["yaxis"]["range"]
                    for path in path_set:
                        for ind in all_data[path]["flags"]:
                            if plot_type == "llapi":
                                x_val = all_data[path]["llapi"]["time"][ind[1]]
                            else:
                                x_val = all_data[path]["time"][ind[0]]
                            fig["props"]["figure"]["data"].append({
                                            "x": [x_val, x_val],
                                            "y": y_range,
                                            "line": {"dash": path_dash_style[path], "color": "black"},
                                            "showlegend":False,
                                            "name": "flag",
                                            "meta":[path, plot_type]})
        return plots
    else:
        for fig in plots:
            if "data" in fig["props"]["figure"].keys():
                temp_plots = deepcopy(fig["props"]["figure"]["data"])
                for plot in temp_plots:
                    if "name" in plot.keys() and plot["name"] == "flag":
                        fig["props"]["figure"]["data"].remove(plot)
        return plots

# Callback to update plots when user clicks on sidebar buttons
@callback(
    Output({"type": "plot", "index": dash.MATCH}, "style"),
    Output({"type": "plot", "index": dash.MATCH}, "figure"),
    Input({'type': 'plot-checklist', 'index': dash.MATCH}, 'value'),
    Input('load-paths-checklist', 'value'),
    State({'type': 'plot', 'index': dash.MATCH}, 'figure'),
    State('all-data', 'data'),
    State('path-dash-style', 'data'),
    State('input-color', 'data'),
    State('output-color', 'data'),
    State('llapi-color', 'data'),
    State("flags", "value"),
    prevent_initial_call=True
)
def update_graph_show(data_type_checklist, path_name, fig, all_data, path_dash_style, input_colors, output_colors, llapi_colors, flag):
    # Based on the current selected load-paths and the plot-checklist, form dictionary of all data
    # that needs to be plotted. Dict will be {path-name: data-type}
    to_plot = {path: deepcopy(data_type_checklist) for path in path_name}
    if dash.callback_context.inputs_list[0]["value"] == []:
        return {"display": "none"}, fig

    if to_plot:
        # Check if plot names have plot type in them already. If not, need to grab from callback
        # context inputs and add to name
        for key, val in to_plot.items():
            for i, data_type in enumerate(val):
                split = data_type.split("/")
                if split[0] not in all_data[key].keys(): # plot type not in name
                    for callback_input in dash.callback_context.inputs_list:
                        if data_type in callback_input["value"]:
                            plot_type = callback_input["id"]["index"]
                            val[i] = plot_type + "/" + data_type
    else:
        return {"display": "none"}, fig

    if flag:
        do_flag = True
    else:
        do_flag = False
    adjust_flag_range = False
    if "data" in fig.keys():
        # Remove plots that were unchecked
        temp_data = deepcopy(fig["data"])
        for data in temp_data:
            if data["meta"][0] not in ["legend"]:
                if data["meta"][0] not in to_plot.keys():
                    fig["data"].remove(data)
                elif data["meta"][1] + "/" + data["name"] not in to_plot[data["meta"][0]] and data["name"] != "flag":
                    fig["data"].remove(data)
        # Disinclude plots that have already been made
        for data in fig["data"]:
            if data["meta"][0] in to_plot.keys() and data["meta"][1] + "/" + data["name"] in to_plot[data["meta"][0]]:
                to_plot[data["meta"][0]].remove(data["meta"][1] + "/" + data["name"])
                if not to_plot[data["meta"][0]]:
                    del to_plot[data["meta"][0]]
        # If flag plot already exists, don't add again, but need to adjust the y range
        for data in fig["data"]:
            if do_flag and data["name"] == "flag" and data["meta"][0] in to_plot.keys():
                do_flag = False
                adjust_flag_range = True

    plots = []
    for key, val in to_plot.items():
        time = all_data[key]["time"]
        for data_type in val:
            split = data_type.split("/")
            data_type = split[0]
            if "llapi" == data_type:
                time = all_data[key]["llapi"]["time"]
            if len(split) > 2:  # More than 2 splits, is llapi data
                sub_type = "/".join(split[1:3])
                joint = "/".join(split[3:])
            else:
                sub_type = None
                joint = split[1]
            if isinstance(all_data[key][data_type], dict):
                if sub_type:
                    data = all_data[key][data_type][sub_type][joint]
                else:
                    data = all_data[key][data_type][joint]
            else:
                data = all_data[key][data_type]
            meta_data_type = data_type
            if data_type == "input":
                color = input_colors[joint]
            elif data_type == "output":
                color = output_colors[joint]
            elif data_type == "llapi":
                if sub_type:
                    meta_data_type += "/" + sub_type
                    color = llapi_colors[sub_type][joint]
                else:
                    color = llapi_colors[joint]
            else:
                color = "#3283FE"
            plots.append({"x": time,
                          "y": data,
                          "name":joint,
                          "line": {"dash": path_dash_style[key],
                          "color": color},
                          "showlegend":False,
                          "meta":[key, meta_data_type]})
            # Add flag lines if flag is true and flag lines for this plot has not been added yet
            if do_flag:
                min_y = min(data)
                max_y = max(data)
                if "data" in fig.keys() and "yaxis" in fig["layout"].keys() and "range" in fig["layout"]["yaxis"].keys():
                    y_range = fig["layout"]["yaxis"]["range"]
                else:
                    y_range = [min_y, max_y]

                if min_y < y_range[0]:
                    y_range[0] = min_y
                if max_y > y_range[1]:
                    y_range[1] = max_y
                for ind in all_data[key]["flags"]:
                    if "llapi" == data_type:
                        x_val = time[ind[1]]
                    else:
                        x_val = time[ind[0]]
                    plots.append({"x": [x_val, x_val],
                                  "y": y_range,
                                  "line": {"dash": path_dash_style[key], "color": "black"},
                                  "showlegend": False,
                                  "name": "flag",
                                  "meta":[key, meta_data_type]})
            # Adjust y values of flag lines to account for new data
            if adjust_flag_range:
                min_y = min(data)
                max_y = max(data)
                if "yaxis" in fig["layout"].keys() and "range" in fig["layout"]["yaxis"].keys():
                    y_range = fig["layout"]["yaxis"]["range"]
                else:
                    y_range = [min_y, max_y]

                if min_y < y_range[0]:
                    y_range[0] = min_y
                if max_y > y_range[1]:
                    y_range[1] = max_y
                for data in fig["data"]:
                    if data["name"] == "flag":
                        data["y"] = y_range

    # Add/remove dummy plots for legend. There's probably a better way to do this without the
    # multiple loops
    if plots:
        legend_names = set()
        path_legend_names = set()
        for data in data_type_checklist:
            data_split = data.split("/")
            if len(data_split) < 2:
                legend_names.add(data_split[0])
            elif "-" in data_split[0]:
                legend_names.add(data)
            else:
                legend_names.add(data.split("/")[1])
        if data_type_checklist:
            for data in path_name:
                path_legend_names.add(data)

        # Loop through all current plots to find already existing legends (don't need to plot, can
        # remove from legend_names) and unneeded legends (remove from fig["data"])
        if "data" in fig.keys():
            for plot in temp_data:
                if plot["meta"][0] == "legend":
                    # Legend plot not needed, remove
                    if plot["name"] not in legend_names and plot["name"] not in legend_names:
                        fig["data"].remove(plot)
                    elif plot["name"] in legend_names: # Legend plot already exists, remove from legend_names
                        legend_names.remove(plot["name"])
                    elif plot["name"] in path_legend_names:
                        path_legend_names.remove(plot["name"])
        # Create the new legends
        for name in legend_names:
            if data_type_checklist:
                type_split = data_type_checklist[0].split("/")
                if type_split[0] == "input":
                    color = input_colors[name]
                elif type_split[0] == "output":
                    color = output_colors[name]
                elif data_type == "llapi":
                    if sub_type:
                        color = llapi_colors[sub_type][joint]
                    else:
                        color = llapi_colors[joint]
                else:
                    color = "#3283FE"
                plots.append({"x": [None],
                              "y": [None],
                              "name": name,
                              "line": {"dash": "solid", "color": color},
                              "showlegend":True,
                              "meta":["legend", name]})
        for name in path_legend_names:
            newline = len(name) // 20
            if newline > 0:
                name_wrap = ""
                for i in range(newline):
                    name_wrap += name[20*i:20*(i+1)] + "<br>"
                name_wrap = name_wrap[:-4]
            else:
                name_wrap = name
            plots.append({"x": [None],
                          "y": [None],
                          "name": name_wrap,
                          "line": {"dash": path_dash_style[name], "color": "black"},
                          "showlegend":True,
                          "meta":["legend", name]})

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=8050, type=int)
    args = parser.parse_args()

    app.run_server(debug=False, port=args.port)