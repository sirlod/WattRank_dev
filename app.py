# -*- coding: utf-8 -*-
"""
Created on Sun May  1 13:21:11 2022.

@author: Marcin Orzech
"""

# Import Python Libraries
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
import plotly.express as px
import numpy as np
import database
import cell_energy


config = {"displaylogo": False}
HOVER_DATA_DICT = {
    "Specific Energy (Wh/kg)": False,
    "Specific Power (W/kg)": True,
    "Specific Power - Peak (W/kg)": True,
    "Energy density (Wh/L)": False,
    "Average Voltage (V)": True,
    "Internal resistance (mOhm)": True,
    "C rate (discharge)": True,
    "C rate (charge)": True,
    "Capacity (Ah)": True,
    "Energy (Wh)": True,
    "Technology": False,
    "Category": True,
    "Cathode": True,
    "Anode": True,
    "Electrolyte": True,
    "Form factor": True,
    "Cycle life": True,
    "Measurement temperature": True,
    "Publication date": True,
    "Maturity": True,
    "Additional tags": True,
    "Reference/source": False,
}


def read_file(name):
    with open(name, "r") as file:
        text = file.read()
    return text


def page_config():
    """Setups page settings and menu options.
    Must be called as first streamlit command."""
    st.set_page_config(
        page_title="WattRank",
        page_icon="⚡",
        layout="wide",
        menu_items={
            "Get Help": None,
            "Report a bug": "mailto:wattrank@gmail.com",
            "About": "## WattRank is still under construction. v0.0.2",
        },
    )


def session_state_init(name):
    """Initialize sesion state counting for streamlit functions."""
    if name not in st.session_state:
        st.session_state[name] = 0


def reset_state(name):
    st.session_state[name] += 1


def layout():
    """
    Modify CSS layout.

    Returns
    -------
    None.

    """

    # Hide footer and hamburger menu
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def read_sql(table_name):
    if table_name not in st.session_state:
        st.session_state[table_name] = database.get_table(table_name)
    return st.session_state[table_name]


def read_csv(path):  # depreciated
    """
    Read the data and assign to df.

    Parameters
    ----------
    path : str
        path to data file

    Returns
    -------
    dataframe object

    """
    return pd.read_csv(path)


def fetch_latest_data():
    if st.session_state["df_state"] == 0:
        return read_sql("data")
    else:
        return st.session_state.data


def rename_columns(df, df_params):
    name_map = dict(zip(df_params["short_name"], df_params["long_name"]))
    df.rename(columns=name_map, inplace=True)


def replace_nan(dataframe):
    """Replace NaN values with 'None' in non numerical columns."""
    object_cols = dataframe.select_dtypes(["object"]).fillna("_No data_")
    dataframe[object_cols.columns] = object_cols
    return dataframe


def clean_axes_data(data, x, y):
    """
    Drop rows with no x and y values.

    Parameters
    ----------
    data : pd.DataFrame
        input data.
    x, y : str.
        axes column names.

    Returns
    -------
    data : modified pd.DataFrame

    """
    data = data[(data[x].notna()) & (data[y].notna())]
    return data


def scatter_plot(data, x, y, title, group_color, group_symbol, size):
    """
    Construct layout for basic scatter plot.

    Parameters
    ----------
    data : DataFrame
        Data to plot.
    x : float
        Data column for x axis.
    y : float
        Data column for y axis.
    title : str
        Title of the plot.
    group_color : str
        Name of the column the markers are grouped by.
        Passed to color and legend.
        Gets value from the groupby() function.
    group_symbol : str
        Name of the column the markers are differentiated by.
        Passed to symbol.
        Gets value from the groupby() function.
    size : DataSeries
        Column for sizing the markers. Default = None
        Calls size_checkbox() function.

    Returns
    -------
    fig : object
        Plotly fig object.
    """
    # data = clean_axes_data(data, x, y)
    fig = px.scatter(
        data,
        x=x,
        y=y,
        color=group_color,
        symbol=group_symbol,
        height=600,
        title=title,
        hover_name="Name",
        hover_data=HOVER_DATA_DICT,
        size=size,
        size_max=25,
    )
    # make markers without cycle life (=1) visible
    fig.update_traces(marker_sizemin=3) 
    fig.update_xaxes(
        showline=True,
        linewidth=2,
        rangemode="tozero",
        mirror=True,
        title_font_size=20,
        showspikes=True,
        # rangeslider_visible=True,
    )
    fig.update_yaxes(
        showline=True,
        linewidth=2,
        rangemode="tozero",
        mirror=True,
        title_font_size=20,
        showspikes=True,
    )
    fig.update_layout(
        plot_bgcolor="rgba(255,255,255,0)",
        template="simple_white",
        modebar_add=["drawcircle", "drawclosedpath", "eraseshape"],
        modebar_remove=["lasso2d", "select2d", "resetScale2d", "pan"],
        modebar_orientation="h",
        legend_orientation="v",
        legend_y=1,
        legend_borderwidth=2,
        title_font_size=30,
        title_x=0.5,
        title_xref="paper",
        title_xanchor="center",
        title_yanchor="top",
    )

    return fig


def connect_legend_with_clusters(fig):
    """Toggles cluster highlight together with markers."""
    try:
        fig.for_each_trace(
            lambda t: t.update(
                # legendgroup=t.marker.color,
                legendgroup=t.name.split(",")[0],
                legendgrouptitle_text=t.name.split(",")[0],
                name=t.name.split(",")[1],
            )
        )
    except:
        pass
    return fig


def confidence_ellipse(x, y, n_std=1.6, size=100):
    """
    Get the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    size : int
        Number of points defining the ellipse
    Returns
    -------
    String containing an SVG path for the ellipse

    References (H/T)
    ----------------
    https://matplotlib.org/3.1.1/gallery/statistics/confidence_ellipse.html
    https://community.plotly.com/t/arc-shape-with-path/7205/5
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    theta = np.linspace(0, 2 * np.pi, size)
    ellipse_coords = np.column_stack(
        [ell_radius_x * np.cos(theta), ell_radius_y * np.sin(theta)]
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    x_scale = np.sqrt(cov[0, 0]) * n_std
    x_mean = np.mean(x)

    # calculating the stdandard deviation of y ...
    y_scale = np.sqrt(cov[1, 1]) * n_std
    y_mean = np.mean(y)

    translation_matrix = np.tile([x_mean, y_mean], (ellipse_coords.shape[0], 1))
    rotation_matrix = np.array(
        [
            [np.cos(np.pi / 4), np.sin(np.pi / 4)],
            [-np.sin(np.pi / 4), np.cos(np.pi / 4)],
        ]
    )
    scale_matrix = np.array([[x_scale, 0], [0, y_scale]])
    ellipse_coords = (
        ellipse_coords.dot(rotation_matrix).dot(scale_matrix)
        + translation_matrix
    )

    path = f"M {ellipse_coords[0, 0]}, {ellipse_coords[0, 1]}"
    for k in range(1, len(ellipse_coords)):
        path += f"L{ellipse_coords[k, 0]}, {ellipse_coords[k, 1]}"
    path += " Z"
    # return path <- orignal code to return path for ploting shapes
    # returning df for line plot instead of path
    return pd.DataFrame(ellipse_coords)


def highlight_clusters(fig, df, category, x, y):
    """
    Highlight clusters of scatter points.

    Parameters
    ----------
    fig : plotly fig object
        input figure.
    df : DataFrame object
        input data of the plot.
    category : str
        Name of the column the points are grouped by.
        Taken from groupby() func.
    x, y : str
        x and y axes column names.

    Returns
    -------
    fig : plotly fig object
        Modified fig with drawn circles around points clusters.

    """
    category_list = df[category].unique().tolist()
    df = clean_axes_data(df, x, y)

    for label in category_list:
        for d in fig.data:
            if label == d.name.split(",")[0]:
                color = d.marker.color
        coords = confidence_ellipse(
            df.loc[df[category] == label, x], df.loc[df[category] == label, y]
        )
        fig.add_scatter(
            x=coords[0],
            y=coords[1],
            fill="toself",
            legendgroup=label,
            showlegend=False,
            fillcolor=color,
            opacity=0.2,
            line_width=0,
            name=label,
            hoverinfo="skip",
        )
    # fig = connect_legend_with_clusters(fig)
    return fig


def columns_layout(widgets_count):
    """
    Layouts widgets into 4 columns.

    Parameters
    ----------
    widgets_count : int
        Number of widgets.

    Returns
    -------
    col_list : list
        Array, where rows are cols and number of rows == widget_count.

    """
    # setting up layout into 3 columns
    col1, col2, col3, col4 = st.columns(4)
    cols = [col1, col2, col3, col4]

    # generator expresion, makes array row=cols, number of rows=widget_count
    col_list = []
    for i in (x for _ in range(widgets_count) for x in cols):
        col_list.append(i)
    return col_list


def list_unique_values(parameter):
    """
    List all unique text values in the column.

    Parameters
    ----------
    parameter : str
        long name of column.

    Returns
    -------
    val_list : set

    """
    val_list = set(df[parameter].str.split(",").sum())
    val_list = {val.strip() for val in val_list}
    return val_list


def filters(df, x, y, preset):
    """
    Filter the input data.

    Parameters.
    ----------
    df : pd.DataFrame
    Input data.
    x, y : axes columns

    Returns
    -------
    new_df : filtered dataframe

    """
    # df = clean_axes_data(df, x, y)
    new_df = df.copy()
    filters_multiselect = [
        "Technology",
        "Category",
        "Cathode",
        "Anode",
        "Electrolyte",
        "Form factor",
        "Maturity",
        "Additional tags",
    ]
    filters_slider = [
        "Specific Energy (Wh/kg)",
        "Energy density (Wh/L)",
        "Specific Power (W/kg)",
        "Specific Power - Peak (W/kg)",
        "Average Voltage (V)",
        "C rate (discharge)",
        "C rate (charge)",
        "Cycle life",
        "Measurement temperature",
        "Internal resistance (mOhm)",
    ]
    filters_slider = list(set(filters_slider) - set([x, y]))
    # all_filters = filters_multiselect + filters_slider
    # filters_count = len(all_filters)
    st.markdown("## *Filters:*")

    # reseting filters using session state count
    if st.button("Reset to default"):
        reset_state("filters")

    # Layout filters into columns
    # col_list = columns_layout(filters_count)

    # drawing multiselect filters
    for parameter in filters_multiselect:
        # with col_list[filters_multiselect.index(option)]:
        st.write("---")
        options_list = list_unique_values(parameter)
        selected_option = st.multiselect(
            parameter,
            options_list,
            default=preset.get(parameter),
            key=parameter + str(st.session_state.filters),
            help=df_params.loc[
                df_params["long_name"] == parameter, "description"
            ].values[0],
        )
        if len(selected_option) > 0:
            new_df = new_df[
                new_df[parameter].apply(
                    lambda x: any(i.strip() in selected_option
                                  for i in x.split(","))
                )
            ]

    # drawing slider filters
    # col_number = len(filters_multiselect)
    for option in filters_slider:
        min_val = float(df[option].min())
        max_val = float(df[option].max())
        set_range = min_val, max_val
        if preset.get(option):
            set_range = preset.get(option)
        if min_val != max_val:
            # with col_list[col_number]:
            st.write("---")
            selected_range = st.slider(
                option,
                min_val,
                max_val,
                value=set_range,
                step=0.1,
                format="%f",
                key=option + str(st.session_state.filters),
                help=df_params.loc[
                    df_params["long_name"] == option, "description"
                ].values[0],
            )
            # dealing with NaN values
            display_NaN = True
            if pd.isna(new_df[option]).any():
                display_NaN = st.checkbox(
                    f"*Include missing values in **{option}**.*", True
                )
            if not display_NaN:
                new_df = new_df[
                    (new_df[option].between(selected_range[0],
                                            selected_range[1]))
                ]
            else:
                new_df = new_df[
                    (
                        new_df[option].between(selected_range[0],
                                               selected_range[1])
                        | (pd.isna(new_df[option]))
                    )
                ]
        # col_number += 1

    return new_df


def filters_preset():
    preset_filters = {}
    form_factors = df["Form factor"].unique().tolist()
    tags = list_unique_values("Additional tags")
    preset_options = [
        "All data",
        "Cells in research stage",
        "Commercial cells in standard conditions",
        "Automotive packs",
        "Cells in development",
    ]
    st.markdown("### *Filters preset:*")
    selected_preset = st.radio(
        "Presets:", preset_options, horizontal=True,
        label_visibility="collapsed"
    )
    if selected_preset == "Cells in research stage":
        preset_filters = {"Maturity": "Research"}
    elif selected_preset == "Commercial cells in standard conditions":
        preset_filters = {
            "Maturity": "Commercial",
            "Form factor": [
                f
                for f in form_factors
                if f not in ["Coin cell", "Pack", "Pack (Cell-to-Pack)"]
            ],
            "Measurement temperature": (20.0, 31.0),
            "Additional tags": [t for t in tags if t not in ["ageing"]],
        }
    elif selected_preset == "Automotive packs":
        preset_filters = {
            "Maturity": "Commercial",
            "Form factor": ["Pack", "Pack (Cell-to-Pack)"],
            "Additional tags": "EV",
        }
    elif selected_preset == "Cells in development":
        preset_filters = {"Maturity": "Development"}
    return preset_filters


def size_checkbox(df):
    """
    Checkbox to match markers size with cycle life.

    Returns
    -------
    size : pd.DataSeries
        Cycle life column.

    """
    if st.checkbox("Match size of the markers with corresponding cycle life"):
        size = df["Cycle life"]
        return size


def groupby(groupby_type):
    """
    Generate radio widget to select column to group by.

    Parameters
    ----------
    groupby_type : str
        Type of grouping ('colour' or 'symbol').

    Returns
    -------
    selected_group : str
        Column name of selected group.
    """
    groups = [
        "Technology",
        "Category",
        "Cathode",
        "Anode",
        "Electrolyte",
        "Form factor",
    ]
    st.markdown(f"### *Group datapoints {groupby_type} by:*")
    selected_group = st.radio(
        f"**Group data by {groupby_type}:**",
        groups,
        1,
        horizontal=True,
        label_visibility="collapsed",
    )
    return selected_group


def plot_widgets(df):
    c1, c2 = st.columns(2)
    with c1:
        group_color = groupby("colour")
    with c2:
        group_symbol = groupby("symbol")
    df = df.sort_values(by=group_color)
    preset = filters_preset()

    with st.sidebar:
        df = filters(df, x, y, preset)

    size = size_checkbox(df)
    return group_color, group_symbol, df, preset, size


def input_field(parameter):
    # highlight labels of required fields
    required_fields = [
        "Name",
        "Specific Energy (Wh/kg)",
        "Specific Power (W/kg)",
        "Energy density (Wh/L)",
        "Technology",
        "Category",
        "Capacity calculation method",
        "Form factor",
        "Maturity",
        "Reference/source",
    ]
    label = parameter
    if parameter in required_fields:
        label = parameter + ":red[*]"

    help_prompt = df_params.loc[
        df_params["long_name"] == parameter, "description"
    ].values[0]
    # columns with new, unique text values
    if parameter in ["Name", "Reference/source"]:
        value = st.text_input(
            label,
            key="form" + parameter + str(st.session_state.form),
            help=help_prompt,
            placeholder=f"Type in {parameter} (required)",
        )
    # columns with predetermined values to select from. no new values allowed
    elif parameter in ["Capacity calculation method", "Maturity"]:
        options_list = list_unique_values(parameter)
        value = st.multiselect(
            label,
            options_list,
            key="form" + parameter + str(st.session_state.form),
            help=help_prompt,
            max_selections=1,
        )
        value = "".join(value)
    # columns with selectable options, but with possibility to add new values
    elif parameter in [
        "Technology",
        "Category",
        "Cathode",
        "Anode",
        "Electrolyte",
        "Form factor",
    ]:
        options_list = list_unique_values(parameter)
        field = st.empty()
        new_value = st.checkbox(
            "Value not in the list. Add new value",
            key=parameter + str(st.session_state.form),
        )
        value = field.multiselect(
            label,
            options_list,
            key="form" + parameter + str(st.session_state.form),
            help=help_prompt,
            max_selections=1,
            disabled=new_value,
        )
        value = "".join(value)
        if new_value:
            value = field.text_input(
                label, help=help_prompt, placeholder="Type in new value"
            )
    # column with both selectable values and new values
    elif parameter == "Additional tags":
        options_list = set(df[parameter].str.split(",").sum())
        options_list.remove("_No data_")
        value = st.multiselect(
            label,
            options_list,
            key="form" + parameter + str(st.session_state.form),
            help=help_prompt,
        )
        if st.checkbox(
            "Add more tags",
            key=parameter + str(st.session_state.form)
        ):
            new_value = st.text_input(
                label,
                help="Make sure to separate tags with commas (,)",
                placeholder="Separate tags with commas",
            )
            value = value + new_value.split(",")
        value = ",".join(value)
    # columns with integer data
    elif parameter in ["Cycle life", "Publication date"]:
        value = st.number_input(
            label,
            min_value=0,
            key="form" + parameter + str(st.session_state.form),
            help=help_prompt,
        )
        if value == 0 and parameter == "Cycle life":
            value = 1
        elif value == 0:
            value = None
    # columns with float data
    else:
        value = st.number_input(
            label,
            min_value=0.0,
            step=0.1,
            format="%.1f",
            key="form" + parameter + str(st.session_state.form),
            help=help_prompt,
        )
        if value == 0:
            value = None
    return value


def input_form():
    inputs = {}
    # Layout into columns
    col_list = columns_layout(len(df.columns))
    # drawing text input fields
    for parameter in df.columns:
        with col_list[df.columns.get_loc(parameter)]:
            st.write("---")
            value = input_field(parameter)
            inputs[parameter] = value
    return inputs


def values_missing(inputs):
    missing = False
    required_fields = [
        "Name",
        "Specific Energy (Wh/kg)",
        "Specific Power (W/kg)",
        "Energy density (Wh/L)",
        "Technology",
        "Category",
        "Capacity calculation method",
        "Form factor",
        "Maturity",
        "Reference/source",
    ]
    for v in required_fields:
        if not inputs[v]:
            missing = True
    return missing


def check_duplicates(data):
    columns = [
        "Specific Energy (Wh/kg)",
        "Specific Power (W/kg)",
        "Energy density (Wh/L)",
        "Reference/source",
    ]
    duplicates = data.duplicated(subset=columns, keep="last")
    duplicated = data[duplicates].index.tolist()
    if duplicates.any():
        st.error(
            f"It seems that we already have this data point. \
                Please check data row {duplicated} in 'Source data' tab."
        )
        return True
    else:
        return False


def send_data_to_database(data: dict):
    # cols = df_params['short_name'].tolist()
    # cols = ','.join(cols)
    values = list(data.values())
    database.upload_row(values)


def email_prompt():
    c1, c2 = st.columns(2)
    c2.write(
        "**Please add your email, so it will possible to contact you \
        regarding the uploaded data. It won't' be shared with anyone \
        beside the author of this website.**"
    )
    with c1:
        address = st.text_input(
            "***Email address:***:red[*]",
            key=str(st.session_state.form),
            placeholder="example@gmail.com",
        )
    return address


def upload_button(inputs, address):
    uploaded = False
    if st.button("Upload data", type="primary"):
        # check if all required fields are filled
        if values_missing(inputs) or not address:
            return st.error("Please fill in all required fields")
        # check for duplicates
        if check_duplicates(df_updated):
            return st.error("Upload failed.")
        # check if email correct
        if "@" not in address:
            return st.error("Please fill in correct email address")
        # upload to sql
        else:
            try:
                send_data_to_database(inputs)
                database.save_email(address)
            except:
                st.error("Something went wrong with the upload. \
                        Please contact marcin.w.orzech@gmail.com")
            else:
                uploaded = True
                reset_state("form")
    if uploaded:
        # clear data from cache, so the updated table will be loaded
        # st.experimental_memo.clear()
        st.cache_data.clear()
        return st.success("Thank you for sharing!")


# @st.cache
def convert_df(df):
    return df.to_csv().encode("utf-8")


def add_data_to_df(df, new_data):
    new_data = {col: new_data[col] for col in df.columns if col in new_data}
    st.session_state.data = pd.concat(
    [df, pd.DataFrame(new_data, index=[len(df)])], ignore_index=True,
    )
    return st.session_state.data


# def kofi_button():
#     button = """
#         <body>
#         <style>
#             .floatingchat-container-wrap {
#                 position:fixed;
#                 bottom:16px;
#                 left:170px;
#                 z-index:99999999!important;
#                 width:100%;
#                 height:65px;
#                 max-width:180px
#             }
#         </style>
#         <script src='https://storage.ko-fi.com/cdn/scripts/overlay-widget.js'></script>
#             <script>
#             kofiWidgetOverlay.draw('marcinorzech', {
#                 'type': 'floating-chat',
#                 'floating-chat.donateButton.text': 'Support me',
#                 'floating-chat.donateButton.background-color': '#d9534f',
#                 'floating-chat.donateButton.text-color': '#fff',
#             });
#             </script>
#         </body>
#     """
#     return button
def kofi_button():
    button = """
        <body>
        <style>
            .floatingchat-container-wrap {
                position:fixed;
                bottom:16px;
                left:170px;
                z-index:99999999!important;
                width:100%;
                height:65px;
                max-width:180px;
            }
        </style>
        <script src='https://storage.ko-fi.com/cdn/scripts/overlay-widget.js'></script>
            <script>
            kofiWidgetOverlay.draw('marcinorzech', {
                'type': 'floating-chat',
                'floating-chat.donateButton.text': 'Support me',
                'floating-chat.donateButton.background-color': '#d9534f',
                'floating-chat.donateButton.text-color': '#fff',
            });
            </script>
        </body>
    """
    return button

def float_button(button):
    components.html(button, height=650, width=355)

    st.markdown(
        """
        <style>
            iframe[width="355"] {
                position: fixed;
                bottom: 20px;
                right: 50px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


page_config()
layout()

session_state_init("filters")
session_state_init("form")
session_state_init("calc")
session_state_init("df_state")
df = fetch_latest_data()
df_params = read_sql("parameters")
rename_columns(df, df_params)
df = replace_nan(df)

# Multipage menu
with st.sidebar:
    choose = option_menu(
        "WattRank",
        [
            "Home",
            "Energy plots",
            # 'Ragone plot',
            "Custom plot",
            "Add data",
            "Source data",
            "Cell energy calculator",
            "About",
        ],
        icons=[
            "house",
            "battery-full",
            # 'hourglass-split',
            "graph-up",
            "upload",
            "stack",
            "calculator",
            "person lines fill",
        ],
        menu_icon="lightning-charge",
        default_index=0,
        styles={
            "container": {"padding": "6!important"},
            "icon": {"color": "#E5625E", "font-size": "25px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#E6E8E6",
            },
            "nav-link-selected": {"background-color": "#333399"},
        },
    )

if choose == "Home":
    ABOUT = read_file("readme.md")
    st.title("⚡ WattRank")
    st.markdown(ABOUT)
    float_button(kofi_button())
    "---"

elif choose == "Energy plots":
    x = "Specific Energy (Wh/kg)"
    y = "Energy density (Wh/L)"
    y2 = "Specific Power (W/kg)"

    group_color, group_symbol, df, preset, size = plot_widgets(df)

    fig_energy = scatter_plot(df, x, y, f"{y} vs {x}",
                              group_color, group_symbol, size)
    fig_energy = highlight_clusters(fig_energy, df, group_color, x, y)
    fig_energy = connect_legend_with_clusters(fig_energy)

    fig_power = scatter_plot(df, x, y2, f"{y2} vs {x}",
                             group_color, group_symbol, size)
    fig_power = highlight_clusters(fig_power, df, group_color, x, y2)
    fig_power = connect_legend_with_clusters(fig_power)

    st.plotly_chart(fig_energy, use_container_width=True,
                    theme=None, config=config)
    st.plotly_chart(fig_power, use_container_width=True,
                    theme=None, config=config)

# elif choose == 'Ragone plot':
#     st.write('Work in progress...')

elif choose == "Custom plot":
    axes_options = df.columns.drop(["Additional tags", "Reference/source"])
    c1, c2 = st.columns(2)
    with c1:
        x = st.selectbox("X axis", axes_options)
    with c2:
        y = st.selectbox("Y axis", axes_options)

    if x == "Name" or y == "Name":
        st.info("Select axes values")
    elif x == y:
        st.error("The value for X and Y axes cannot be the same")
    else:
        group_color, group_symbol, df, preset, size = plot_widgets(df)

        fig_custom = scatter_plot(
            df, x, y, f"{y} vs {x}", group_color, group_symbol, size
        )

        if st.checkbox("Hihlight clusters"):
            fig_custom = highlight_clusters(fig_custom, df, group_color, x, y)

        fig_custom = connect_legend_with_clusters(fig_custom)
        fig_custom.update_xaxes(rangemode="nonnegative")

        st.plotly_chart(fig_custom, use_container_width=True,
                        theme=None, config=config)

elif choose == "Add data":
    st.write("## Upload your own data:")
    new_data = input_form()
    df_updated = pd.concat(
        [df, pd.DataFrame(new_data, index=[len(df)])], ignore_index=True
    )
    "---"
    if st.button("Clear all"):
        reset_state("form")

    st.write("### Your new data:")
    st.dataframe(
        df_updated.tail(3)
        .style.format(thousands="", precision=1)
        .apply(
            lambda x: [
                "background-color: #8587BD" if i == df_updated.index[-1]
                else "" for i in x.index
            ],
            axis=0,
        )
    )

    st.info(
        "***If everything looks ok and you are sure it is correct, \
            click below to upload the data to server.***"
    )
    address = email_prompt()
    upload_button(new_data, address)

elif choose == "Source data":
    st.title("WattRank data:")
    st.dataframe(df.style.format(thousands="", precision=1))
    st.download_button(
        label="***Download data as .csv***",
        data=convert_df(df),
        file_name="WattRank.csv",
        mime="text/csv",
    )
    "---"
    st.markdown("## Parameters description:")
    st.dataframe(
        df_params[["long_name", "description"]].set_index("long_name"),
        use_container_width=True,
    )
    "---"
    st.markdown(
        """
                ### Additional references and acknowledgments
                - Data points 29-74 were aggregated from Enpolite data - 
                https://enpolite.org/ and *P. Dechent, A. Epp, D. Jöst, 
                Y. Preger, P. M. Attia, W. Li, and D. U. Sauer, 
                ENPOLITE: Comparing Lithium-Ion Cells across Energy, Power, 
                Lifetime, and Temperature, 
                ACS Energy Letters 2021 6 (6), 2351-2355*   
                
                - Data points 75-142 were collected from Matt Lacey's website 
                (http://lacey.se/) and 
                *Frith, J.T., Lacey, M.J. & Ulissi, U. 
                A non-academic perspective on the future of 
                lithium-based batteries. 
                Nat Commun 14, 420 (2023).*
                """
    )

elif choose == "Cell energy calculator":
    calc_data = cell_energy.run_calc()
    if calc_data:
        df = add_data_to_df(df, calc_data)
        st.dataframe(
            df.tail(st.session_state.df_state+1)
            .style.format(thousands="", precision=1)
            .apply(
                lambda x: [
                    "background-color: #8587BD" if i == df.index[-1]
                    else "" for i in x.index
                ],
                axis=0,
            )
        )
        reset_state("df_state")
        st.download_button(
            label="***Download calculation results***",
            data=convert_df(df.tail(st.session_state.df_state)),
            file_name="WattRank.csv",
            mime="text/csv",
        )
    if st.button("Clean all calculation results"):
        st.session_state.df_state = 0

elif choose == "About":
    st.title("Hi!")
    c1, c2 = st.columns([2, 1], gap="large")
    with c2:
        st.image(
            "https://avatars.githubusercontent.com/u/103963175?s=400&u=817001116755a3c8370851e82ed54022a575e0b8&v=4"
        )
        st.markdown(
            """
            To contact me regarding the Wattrank or anything else email me at:  
            ✉ marcin.w.orzech@gmail.com
            
            or message on [LinkedIn](https://www.linkedin.com/in/marcin-orzech/)
            """
        )
        st.info(
        """
        
        I develop this website in my free time and aim to keep it free to use
        for all and with full access to the data. If you find Wattrank useful and 
        want to support its development or just want to show apprieciation, 
        please consider buying me a cup of coffee. It will keep me motivated 
        and help with the running costs of this site!
        
        [![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/E1E3OIB2R)
        
        """
        )

    with c1:
        st.markdown(
            """
                I'm Marcin, originally from northern Poland but now residing in Norrköping, Sweden. I work as an R&D manager at Redox.me, a small company that specializes in creating electrochemical cells for research and materials development. I’m overseeing a range of projects, but my main focus is on designing battery test cells for various in situ measurements.  
                
                I completed an Engineering Doctorate (EngD) at Swansea University, where I spent 4 years researching Na-ion batteries for stationary energy storage systems (ESS). I hold two bachelor's degrees - in Materials Engineering and Mechatronics, as well as a Master's degree in Materials Engineering. My passion for batteries started during my Master's program, where for the thesis I tried my chances at making Li-ion cells with SiOx anodes mixed with graphene made in a kitchen blender.  
                
                Wattrank came to be as a fun project for me to dive into Python and SQL. I wanted to turn my learning experience into something helpful for others, and so Wattrank was born! I'm always open to feedback and suggestions on how to improve the site. I'm constantly working to improve the technical side of the website and add more data to the plots.  
                
                My free time I devote to my family or risk minor injuries while rock climbing. To unwind, I love playing board games with friends, sometimes with a glass of homemade beer. 

                """
        )

        st.markdown(
            """
                    ### Licensing and citing
                    The content of this project itself is licensed under the
                    [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/),
                    and the underlying source code used to format
                    and display that content is licensed under the
                    [GPL-3.0 license](https://github.com/sirlod/WattRank_dev/blob/main/LICENSE).
                    
                    If you are using Wattrank in your work, please cite  as (or accordingly to your prefered citation style):  
                    Orzech, M. W.; *WattRank - Compare energy storage devices.*; WattRank.; https://wattrank.com/  (access date: )
                    """
        )
