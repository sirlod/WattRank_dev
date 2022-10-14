# -*- coding: utf-8 -*-
"""
Created on Sun May  1 13:21:11 2022.

@author: Marcin Orzech
"""

# Import Python Libraries
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
# import plotly as py
import plotly.express as px
import numpy as np


config = {'displaylogo': False}
HOVER_DATA_DICT = {
            'Specific Energy (Wh/kg)': False,
            'Specific Power (W/kg)': True,
            'Energy density (Wh/L)': False,
            'Average OCV': True,
            'C rate (discharge)': True,
            'C rate (charge)': True,
            'Technology': True,
            'Category': False,
            'Cathode': True,
            'Anode': True,
            'Electrolyte': True,
            'Form factor': True,
            'Cycle life': True,
            'Measurement temperature': True,
            'Publication date': True,
            'Maturity': True,
            'Reference/source': True,
            }


def read_file(name):
    with open(name, 'r') as file:
        text = file.read()
    return text


def page_config():
    """Setups page settings and menu options. Must be called as first streamlit command."""
    st.set_page_config(
          page_title="WattRank",
          page_icon="⚡",
          layout="wide",
          menu_items={
              # 'Get Help': 'https://www.extremelycoolapp.com/help',
              # 'Report a bug': "https://www.extremelycoolapp.com/bug",
              'About': "## WattRank is still under construction. v0.01"
          }
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
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;} </style>', unsafe_allow_html=True)
    # st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{padding-left:2px;padding-bottom:1px;}</style>', unsafe_allow_html=True)
    st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{padding-right:10px; padding-bottom:2px;} div.st-bx{margin-right:10px} div.st-bo{font-size:20px;font-weight:bold}</style>', unsafe_allow_html=True)
    st.write('<style>label.css-qrbaxs.effi0qh3{font-size:16px;font-weight:bold}</style>', unsafe_allow_html=True)

    # Hide footer and hamburger menu
    hide_streamlit_style = """
                <style>
                MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


@st.cache(allow_output_mutation=True)
def read_csv(path):
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


def replace_nan(dataframe):
    """Replace NaN values with 'None' in non numerical columns."""
    df.select_dtypes('object').fillna('None', inplace=True)
    return df


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


def scatter_plot(data, x, y, title, group, size):
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
    group : str
        Name of the column the markers are grouped by.
        Passed to color and legend.
        Gets value from the groupby() function.
    size : DataSeries
        Column for sizing the markers. Default = None
        Calls size_checkbox() function.

    Returns
    -------
    fig : object
        Plotly fig object.
    """
    data = clean_axes_data(data, x, y)
    fig = px.scatter(data,
                     x=x,
                     y=y,
                     color=group,
                     height=600,
                     title=title,
                     hover_name='Name',
                     hover_data=HOVER_DATA_DICT,
                     size=size,
                     size_max=25
                     )
    fig.update_traces(marker_sizemin=3)  # make markers without cycle life (=1) visible
    fig.update_xaxes(showline=True,
                     linewidth=2,
                     rangemode="tozero",
                     mirror=True,
                     title_font_size=20,
                     showspikes=True,
                     )
    fig.update_yaxes(showline=True,
                     linewidth=2,
                     rangemode="tozero",
                     mirror=True,
                     title_font_size=20,
                     showspikes=True,
                     )
    fig.update_layout(
        plot_bgcolor='rgba(255,255,255,0)',
        template='simple_white',
        modebar_add=['drawcircle', 'drawclosedpath', 'eraseshape'],
        modebar_remove=['lasso2d', 'select2d', 'resetScale2d', 'pan'],
        modebar_orientation='h',
        legend_orientation='v',
        legend_y=1,
        legend_borderwidth=2,
        title_font_size=30,
        title_x=0.5,
        title_xref='paper',
        title_xanchor='center',
        title_yanchor='top',
        )
    fig = connect_legend_with_clusters(fig)

    return fig


def connect_legend_with_clusters(fig):
    """Click legend to toggle cluster highlighting together with markers."""
    group_list = df[groupby].unique().tolist()
    for label in group_list:
        fig.update_traces(selector=label,
                          legendgroup=label,
                          )
    return fig


def confidence_ellipse(x, y, n_std=1.9, size=100):
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
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    theta = np.linspace(0, 2 * np.pi, size)
    ellipse_coords = np.column_stack([ell_radius_x * np.cos(theta), ell_radius_y * np.sin(theta)])

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    x_scale = np.sqrt(cov[0, 0]) * n_std
    x_mean = np.mean(x)

    # calculating the stdandard deviation of y ...
    y_scale = np.sqrt(cov[1, 1]) * n_std
    y_mean = np.mean(y)

    translation_matrix = np.tile([x_mean, y_mean], (ellipse_coords.shape[0], 1))
    rotation_matrix = np.array([[np.cos(np.pi / 4), np.sin(np.pi / 4)],
                                [-np.sin(np.pi / 4), np.cos(np.pi / 4)]])
    scale_matrix = np.array([[x_scale, 0],
                            [0, y_scale]])
    ellipse_coords = ellipse_coords.dot(rotation_matrix).dot(scale_matrix) + translation_matrix

    path = f'M {ellipse_coords[0, 0]}, {ellipse_coords[0, 1]}'
    for k in range(1, len(ellipse_coords)):
        path += f'L{ellipse_coords[k, 0]}, {ellipse_coords[k, 1]}'
    path += ' Z'
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

    for label in category_list:
        color = fig.data[category_list.index(label)].marker.color
        coords = confidence_ellipse(
            df.loc[df[category] == label, x],
            df.loc[df[category] == label, y])
        fig.add_scatter(
                  x=coords[0],
                  y=coords[1],
                  fill='toself',
                  legendgroup=label,
                  showlegend=False,
                  fillcolor=color,
                  opacity=0.2,
                  line_width=0,
                  name=label,
                  hoverinfo='skip',
                  )
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
    col1, blank1, col2, blank2, col3, blank3, col4 = st.columns([6, 1, 6, 1, 6, 1, 6])
    cols = [col1, col2, col3, col4]

    # generator expresion, makes array row=cols, number of rows=widget_count
    col_list = []
    for i in (x for _ in range(widgets_count) for x in cols):
        col_list.append(i)
    return col_list


def filters(df, x, y):
    """
    Filter the input data.

    Parameters.
    ----------
    df : pd.DataFrame
    Input data.

    Returns
    -------
    new_df : filtered dataframe

    """
    df = clean_axes_data(df, x, y)
    new_df = df.copy()
    filters_multiselect = ['Technology', 'Category', 'Cathode', 'Anode', 'Electrolyte', 'Form factor', 'Maturity', 'Additional tags']
    filters_slider = ['Specific Energy (Wh/kg)', 'Energy density (Wh/L)', 'Specific Power (W/kg)', 'Average OCV', 'C rate (discharge)', 'C rate (charge)', 'Cycle life', 'Measurement temperature']
    filters_slider = list(set(filters_slider)-set([x, y]))
    all_filters = filters_multiselect + filters_slider
    filters_count = len(all_filters)

    # reseting filters using session state count

    if st.button("Reset to default"):
        reset_state('filters')

    # Layout filters into columns
    col_list = columns_layout(filters_count)

    # drawing multiselect filters
    for option in filters_multiselect:
        with col_list[filters_multiselect.index(option)]:
            options_list = df[option].dropna().unique().tolist()
            selected_option = st.multiselect(option, options_list, key=option + str(st.session_state.filters))
            if len(selected_option) > 0:
                new_df = new_df[(new_df[option].isin(selected_option))]

    # drawing slider filters
    col_number = len(filters_multiselect)
    for option in filters_slider:
        min_val = float(df[option].min())
        max_val = float(df[option].max())
        if min_val != max_val:
            with col_list[col_number]:
                selected_range = st.slider(option, min_val, max_val, (min_val, max_val), 0.1, '%f', key=option + str(st.session_state.filters))
                # dealing with NaN values
                display_NaN = True
                if pd.isna(new_df[option]).any():
                    display_NaN = st.checkbox(f'Some values in {option} are missing, include them?', True)
                if not display_NaN:
                    new_df = new_df[(new_df[option].between(selected_range[0], selected_range[1]))]
                else:
                    new_df = new_df[(new_df[option].between(selected_range[0], selected_range[1]) | (pd.isna(new_df[option])))]
            col_number += 1

    return new_df


def size_checkbox():
    """
    Checkbox to match markers size with cycle life.

    Returns
    -------
    size : pd.DataSeries
        Cycle life column.

    """
    if st.checkbox('Match size of the markers with corresponding cycle life'):
        size = df['Cycle life']
        return size


def groupby():
    """
    Generate radio widget to select column to group by.

    Returns
    -------
    selected_group : str
        column name of selected group.

    """
    groups = ['Technology', 'Category', 'Cathode', 'Anode', 'Electrolyte', 'Form factor']
    selected_group = st.radio('Group data by:', groups, 1)
    return selected_group


page_config()
layout()

df = read_csv('data.csv')
df = replace_nan(df)
session_state_init('filters')

# Multipage menu
with st.sidebar:
    choose = option_menu(
        'WattRank',
        ['About',
         'Energy plot',
         'Ragone plot',
         'Custom plot',
         'Add data',
         'Source data',
         'Contact'],
        icons=['house',
               'battery-full',
               'hourglass-split',
               'graph-up',
               'upload',
               'stack',
               'person lines fill'],
        menu_icon='lightning-charge',
        default_index=0,
        styles={
               'container': {'padding': '6!important'},
               'icon': {'color': '#E5625E',
                        'font-size': '25px'},
               'nav-link': {'font-size': '16px',
                            'text-align': 'left',
                            'margin': '0px',
                            '--hover-color': '#E6E8E6'},
               'nav-link-selected': {'background-color': '#333399'},
               }
                        )

if choose == 'About':
    ABOUT = read_file('readme.md')
    st.title('⚡ WattRank')
    st.markdown(ABOUT)

elif choose == 'Energy plot':
    groupby = groupby()
    x = 'Energy density (Wh/L)'
    y = 'Specific Energy (Wh/kg)'
    title = 'Gravimetric vs Volumetric Energy'

    with st.expander('Filters'):
        df = filters(df, x, y)

    fig_energy = scatter_plot(df, x, y, title, groupby, size_checkbox())
    fig_energy = highlight_clusters(fig_energy, df, groupby, x, y)
    plot = st.container()

    plot.plotly_chart(fig_energy, use_container_width=True, config=config)

elif choose == 'Ragone plot':
    st.write('Work in progress...')

elif choose == 'Custom plot':
    axes_options = df.columns.drop(['Additional tags', 'Reference/source'])
    c1, c2 = st.columns(2)
    with c1:
        x = st.selectbox('X axis', axes_options)
    with c2:
        y = st.selectbox('Y axis', axes_options)

    if x == 'Name' or y == 'Name':
        st.info('Select axes values')
    elif x == y:
        st.error('The value for X and Y axes cannot be the same')
    else:
        groupby = groupby()
        with st.expander('Filters'):
            df = filters(df, x, y)
        fig_custom = scatter_plot(df, x, y, f'{y} vs {x}', groupby, size_checkbox())
        if st.checkbox('Hihlight clusters'):
            fig_custom = highlight_clusters(fig_custom, df, groupby, x, y)

        plot = st.container()

        plot.plotly_chart(fig_custom, use_container_width=True, config=config)

elif choose == 'Add data':
    st.write('Work in progress...')

elif choose == 'Source data':
    # st.write('Work in progress...')
    st.dataframe(df)

elif choose == "Contact":
    # st.markdown(""" <style> .font {
    # font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;}
    # </style> """, unsafe_allow_html=True)
    st.write('Work in progress...')
    st.write('Please contact: marcin.w.orzech@gmail.com')
    # st.header('Contact Form')
    # with st.form(key='columns_in_form2',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
    #     #st.write('Please help us improve!')
    #     Name=st.text_input(label='Please Enter Your Name') #Collect user feedback
    #     Email=st.text_input(label='Please Enter Email') #Collect user feedback
    #     Message=st.text_input(label='Please Enter Your Message') #Collect user feedback
    #     submitted = st.form_submit_button('Submit')
    #     if submitted:
    #         st.success('Thanks for your feedback!')
