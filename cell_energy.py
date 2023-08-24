# -*- coding: utf-8 -*-
"""
Created on 12 August 2023.
Version 1.0
@authors: Matt Lacey, Marcin Orzech
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px


def lossfrac(Q_p, Q_n, l_p, l_n):
    C_cell = min(Q_p, Q_n)
    Q_max = min(Q_p * (1 + l_p), Q_n * (1 + l_n))

    Q_n_100 = Q_max / (Q_n * (1 + l_n)) * Q_n
    Q_p_100 = Q_max / (Q_p * (1 + l_p)) * Q_p

    Q_n_new = min(Q_n, Q_n_100)
    Q_p_new = min(Q_p, Q_p_100)
    C_cell_new = min(Q_n_new, Q_p_new)

    loss_frac = 1 - (C_cell_new / C_cell)

    return loss_frac


def cyl_dim(diameter, height, void_diameter, can_thickness,
            stack_thickness, headspace):
    # This calculates electrode area using the Archimedes spiral formula

    # Calculate the number of turns in the jelly roll
    turns = (
        (diameter - (2 * can_thickness) - stack_thickness - void_diameter) / 2
    ) / stack_thickness

    # Create a DataFrame of theta steps and the difference between them, dtheta
    theta_values = np.linspace(0, turns * 2 * math.pi, num=2000)
    df = pd.DataFrame({"theta": theta_values})

    # Numerical integration using Archimedes spiral
    df["dtheta"] = [0] + np.diff(df["theta"]).tolist()
    df["r"] = (void_diameter / 2) + (stack_thickness * df["theta"]
                                     / (2 * math.pi))
    df["increment"] = np.sqrt(
        ((void_diameter / 2) + (stack_thickness * df["theta"]
                                / (2 * math.pi))) ** 2 + (stack_thickness
                                                          / (2 * math.pi)) ** 2
    )
    df["length"] = np.cumsum(df["increment"] * df["dtheta"])

    # Return number of turns, jelly roll length, and area
    return {
        "df": df,
        "turns": turns,
        "length": df["length"].iloc[-1],
        "area": df["length"].iloc[-1] * (height - headspace),
    }


def cyl_can_weight(diameter, height, can_thickness, density, extra_mass):
    # Volume of the can itself
    vol = (
        (
            (math.pi * (diameter / 2) ** 2)
            - (math.pi * (diameter / 2 - can_thickness) ** 2)
        )
        * height
    ) + (2 * math.pi * (diameter / 2) ** 2 * can_thickness)

    # Weight is given by
    wt = (vol * density) + extra_mass

    return wt


def cell_param(celltype):
    if celltype == "18650":
        cell = {
            "cell_diameter": 1.8,  # in cm
            "cell_height": 6.5,  # in cm
            "cell_can_thickness": 0.017,  # in cm
            "cell_can_density": 7.9,  # 7.9 for typical steel
            "cell_void_diameter": 0.15,  # empty gap in the center of can, cm
            "cell_headspace": 0.5,  # headspace above jellyroll
            "cell_extra_mass": 2,  # grams
        }
    else:
        cell = {
            "cell_diameter": 2.1,
            "cell_height": 7.0,
            "cell_can_thickness": 0.017,
            "cell_can_density": 7.9,
            "cell_void_diameter": 0.2,
            "cell_headspace": 0.5,
            "cell_extra_mass": 3,
        }
    return cell


def cyl_calculate(
    p_specific_cap=200,
    p_areal_cap=4,
    p_massfrac=0.92,
    p_density=3.4,
    p_nominalV=3.8,
    p_firstloss=10,
    p_arealR=10,
    s_thickness=25,
    s_porosity=0.44,
    s_bulkdensity=0.855,
    e_ratio=2,
    e_density=1.22,
    n_specific_cap=350,
    n_areal_cap=4.4,
    n_massfrac=0.96,
    n_density=1.7,
    n_nominalV=0.1,
    n_firstloss=10,
    n_arealR=10,
    p_cc_thickness=14,
    n_cc_thickness=14,
    p_cc_density=2.7,
    n_cc_density=8.95,
    cell_diameter=1.8,
    cell_height=6.5,
    cell_can_thickness=0.02,
    cell_can_density=7.9,
    cell_void_diameter=0.01,
    cell_headspace=0.5,
    cell_extra_mass=3,
):
    # Definition of quantities into dictionaries
    p = {
        "specific_cap": p_specific_cap,
        "areal_cap": p_areal_cap,
        "massfrac": p_massfrac,
        "density": p_density,
        "nominalV": p_nominalV,
        "firstloss": p_firstloss,
        "arealR": p_arealR,
    }

    n = {
        "specific_cap": n_specific_cap,
        "areal_cap": n_areal_cap,
        "massfrac": n_massfrac,
        "density": n_density,
        "nominalV": n_nominalV,
        "firstloss": n_firstloss,
        "arealR": n_arealR,
    }

    s = {
        "thickness": s_thickness,
        "porosity": s_porosity,
        "bulkdensity": s_bulkdensity
    }

    e = {"ratio": e_ratio, "density": e_density}

    p["cc_thickness"] = p_cc_thickness
    p["cc_density"] = p_cc_density
    n["cc_thickness"] = n_cc_thickness
    n["cc_density"] = n_cc_density

    cell = {
        "diameter": cell_diameter,
        "height": cell_height,
        "can_thickness": cell_can_thickness,
        "can_density": cell_can_density,
        "void_diameter": cell_void_diameter,
        "headspace": cell_headspace,
        "extra_mass": cell_extra_mass,
    }

    # BEGIN CALCULATIONS

    # Composite electrode masses
    p["comp_mass"] = (p["areal_cap"] / p["specific_cap"]) / p["massfrac"]
    n["comp_mass"] = (n["areal_cap"] / n["specific_cap"]) / n["massfrac"]

    p["comp_thickness"] = p["comp_mass"] / p["density"]
    n["comp_thickness"] = n["comp_mass"] / n["density"]

    # Separator mass
    s["mass"] = s["thickness"] * 1e-4 * (1 - s["porosity"]) * s["bulkdensity"]

    # Current collector mass
    p["cc_mass"] = p["cc_thickness"] * 1e-4 * p["cc_density"]
    n["cc_mass"] = n["cc_thickness"] * 1e-4 * n["cc_density"]

    # Electrolyte mass
    e["mass"] = min(p["areal_cap"],
                    n["areal_cap"]) * (e["ratio"] / 1000) * e["density"]

    # Stack areal mass and volume
    stack_mass_areal = (
        (2 * p["comp_mass"])
        + (2 * n["comp_mass"])
        + (2 * s["mass"])
        + p["cc_mass"]
        + n["cc_mass"]
        + (2 * e["mass"])
    )
    stack_thickness = (
        (2 * p["comp_thickness"])
        + (2 * n["comp_thickness"])
        + (2 * s["thickness"] * 1e-4)
        + (p["cc_thickness"] * 1e-4)
        + (n["cc_thickness"] * 1e-4)
    )

    # Jellyroll
    jellyroll = cyl_dim(
        diameter=cell["diameter"],
        height=cell["height"],
        void_diameter=cell["void_diameter"],
        can_thickness=cell["can_thickness"],
        stack_thickness=stack_thickness,
        headspace=cell["headspace"],
    )
    cell["can_mass"] = cyl_can_weight(
        diameter=cell["diameter"],
        height=cell["height"],
        can_thickness=cell["can_thickness"],
        density=cell["can_density"],
        extra_mass=cell["extra_mass"],
    )

    # First cycle loss
    cell["firstcycleloss"] = lossfrac(
        Q_p=p["areal_cap"],
        Q_n=n["areal_cap"],
        l_p=p["firstloss"] / 100,
        l_n=n["firstloss"] / 100,
    )

    # Cell properties
    cell["capacity"] = (
        0.001
        * 2
        * min(p["areal_cap"], n["areal_cap"])
        * jellyroll["area"]
        * (1 - cell["firstcycleloss"])
    )
    cell["energy"] = cell["capacity"] * (p["nominalV"] - n["nominalV"])

    cell["mass"] = (stack_mass_areal * jellyroll["area"]) + cell["can_mass"]
    cell["energy_grav"] = 1000 * cell["energy"] / cell["mass"]

    cell["energy_vol"] = (
        1000 * cell["energy"]
        / (math.pi * (cell["diameter"] / 2) * cell["height"])
    )

    # Mass summary
    mass_summary = {
        "component": [
            "positive electrode",
            "negative electrode",
            "separator",
            "electrolyte",
            "positive c_c_",
            "negative c_c_",
            "can",
        ],
        "value": [
            2 * p["comp_mass"] * jellyroll["area"],
            2 * n["comp_mass"] * jellyroll["area"],
            2 * s["mass"] * jellyroll["area"],
            2 * e["mass"] * jellyroll["area"],
            p["cc_mass"] * jellyroll["area"],
            n["cc_mass"] * jellyroll["area"],
            cell["can_mass"],
        ],
    }

    out = {
        "p": p,
        "n": n,
        "s": s,
        "e": e,
        "cell": cell,
        "jellyroll": jellyroll,
        "mass_summary": mass_summary,
    }

    return out


def plot_mass(calculated_data):
    st.subheader("Mass breakdown")
    mass_data = pd.DataFrame(calculated_data["mass_summary"])
    mass_data["percentage"] = (100 * mass_data["value"]
                               / mass_data["value"].sum())
    mass_data["y"] = ["" for _ in range(len(mass_data.index))]
    component_order = [
        "positive c_c_",
        "positive electrode",
        "separator",
        "electrolyte",
        "negative electrode",
        "negative c_c_",
        "can",
    ]

    # Calculate the percentage and create the plot
    mass_plot = px.bar(
        mass_data,
        y="y",
        x="percentage",
        orientation="h",
        color="component",
        # title="Mass Fraction",
        labels={"percentage": "Mass Fraction", "y": ""},
        category_orders={"component": component_order},
        height=180,
    )
    # Customize the plot
    mass_plot.update_xaxes(ticksuffix="%")
    mass_plot.update_layout(
        legend_title="Component",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=0.9),
    )

    return mass_plot


def plot_cross_section(calculated_data):
    st.subheader("Cross-section schematic")
    # Define the parameters
    cell_diameter = calculated_data["cell"]["diameter"]
    cell_can_thickness = calculated_data["cell"]["can_thickness"]
    theta = calculated_data["jellyroll"]["df"]["theta"]
    r = calculated_data["jellyroll"]["df"]["r"]

    # Calculate the x and y coordinates for the positive and can paths
    x_positive = 10 * r * np.cos(theta)
    y_positive = -10 * r * np.sin(theta)

    x_outer_can = 10 * (cell_diameter / 2) * np.cos(theta)
    y_outer_can = -10 * (cell_diameter / 2) * np.sin(theta)

    x_inner_can = (10 * ((cell_diameter - 2 * cell_can_thickness) / 2)
                   * np.cos(theta))
    y_inner_can = (-10 * ((cell_diameter - 2 * cell_can_thickness) / 2)
                   * np.sin(theta))

    # Create the figure using Plotly
    fig = px.line(x=x_positive, y=y_positive)
    fig.add_scatter(
        x=x_outer_can, y=y_outer_can, line=dict(color="red"), hoverinfo="none"
    )
    fig.add_scatter(
        x=x_inner_can, y=y_inner_can, line=dict(color="red"), hoverinfo="none"
    )
    fig.update_layout(
        xaxis_title="mm",
        yaxis_title=None,
        showlegend=False,
        xaxis=dict(ticks="outside", tickvals=np.arange(-12, 12, 2)),
        yaxis=dict(showticklabels=False, scaleanchor="x", scaleratio=1),
        # width=360,
        height=320,
        xaxis_range=(-12, 12),
        xaxis_showgrid=True,
        xaxis_showline=True,
        yaxis_showline=True,
        xaxis_mirror=True,
        yaxis_mirror=True,
        margin=dict(l=10, r=10, b=10, t=10),
    )

    return fig


def input_fields():
    col2, col3, col4, col5 = st.columns(4)

    input_dict = {}

    with col2:
        st.markdown("### +ve")
        input_dict["p_specific_cap"] = st.number_input(
            "mAh/g", value=175,
            key="p_specific_cap" + str(st.session_state.calc))
        input_dict["p_areal_cap"] = st.number_input(
            "mAh/cm2", value=3.73,
            key="p_areal_cap" + str(st.session_state.calc))
        input_dict["p_massfrac"] = st.number_input(
            "mass fraction", value=0.94,
            key="p_massfrac" + str(st.session_state.calc))
        input_dict["p_density"] = st.number_input(
            "density, g/cc", value=3.5,
            key="p_density" + str(st.session_state.calc))
        input_dict["p_nominalV"] = st.number_input(
            "average E, V", value=3.75,
            key="p_nominalV" + str(st.session_state.calc))
        input_dict["p_arealR"] = st.number_input(
            "R, Ohm cm2", value=7,
            key="p_arealR" + str(st.session_state.calc))
        input_dict["p_firstloss"] = st.number_input(
            "first cyc. loss %", value=7,
            key="p_firstloss" + str(st.session_state.calc))

    with col3:
        st.markdown("### separator")
        input_dict["s_thickness"] = st.number_input(
            "thickness / um", value=16,
            key="s_thickness" + str(st.session_state.calc))
        input_dict["s_porosity"] = st.number_input(
            "porosity", value=0.44,
            key="s_porosity" + str(st.session_state.calc))
        input_dict["s_bulkdensity"] = st.number_input(
            "bulk density, g/cc", value=0.855,
            key="s_bulkdensity" + str(st.session_state.calc))
        st.markdown("---")
        st.markdown("### electrolyte")
        input_dict["e_ratio"] = st.number_input(
            "electrolyte vol, mL/Ah", value=1.8,
            key="e_ratio" + str(st.session_state.calc))
        input_dict["e_density"] = st.number_input(
            "density, g/cc", value=1.22,
            key="e_density" + str(st.session_state.calc))

    with col4:
        st.markdown("### -ve")
        input_dict["n_specific_cap"] = st.number_input(
            "mAh/g", value=350,
            key="n_specific_cap" + str(st.session_state.calc))
        input_dict["n_areal_cap"] = st.number_input(
            "mAh/cm2", value=4.13,
            key="n_areal_cap" + str(st.session_state.calc))
        input_dict["n_massfrac"] = st.number_input(
            "mass fraction", value=0.954,
            key="n_massfrac" + str(st.session_state.calc))
        input_dict["n_density"] = st.number_input(
            "density, g/cc", value=1.55,
            key="n_density" + str(st.session_state.calc))
        input_dict["n_nominalV"] = st.number_input(
            "average E, V", value=0.1,
            key="n_nominalV" + str(st.session_state.calc))
        input_dict["n_arealR"] = st.number_input(
            "R, Ohm cm2", value=5,
            key="n_arealR" + str(st.session_state.calc))
        input_dict["n_firstloss"] = st.number_input(
            "first cyc. loss %", value=10,
            key="n_firstloss" + str(st.session_state.calc))

    with col5:
        st.markdown("### current coll.")
        input_dict["p_cc_thickness"] = st.number_input(
            "+ve thickness / µm", value=15,
            key="p_cc_thickness" + str(st.session_state.calc))
        st.markdown("---")
        n_cc_type = st.radio("-ve c.c.", ("Cu", "Al"),
                             key=st.session_state.calc)
        if n_cc_type == "Al":
            input_dict["n_cc_density"] = 2.7
        else:
            input_dict["n_cc_density"] = 8.95
        input_dict["n_cc_thickness"] = st.number_input(
            "-ve thickness / µm", value=8,
            key="n_cc_thickness" + str(st.session_state.calc))

    return input_dict


def formatted_results(calculated_data, celltype):
    resistance = (
        1000 *
        (calculated_data['p']['arealR'] + calculated_data['n']['arealR']) /
        calculated_data['jellyroll']['area']
    )
    np_ratio = (
        calculated_data['n']['areal_cap'] / calculated_data['p']['areal_cap'])
    voltage = (
        calculated_data['p']['nominalV'] - calculated_data['n']['nominalV'])

    results = {
        "Capacity (Ah)": calculated_data['cell']['capacity'],
        "mass": calculated_data['cell']['mass'],
        "Energy (Wh)": calculated_data['cell']['energy'],
        "Specific Energy (Wh/kg)": calculated_data['cell']['energy_grav'],
        "Energy density (Wh/L)": calculated_data['cell']['energy_vol'],
        "firstcycleloss": calculated_data['cell']['firstcycleloss'],
        "Internal resistance (mOhm)": resistance,
        "turns": calculated_data['jellyroll']['turns'],
        "length": calculated_data['jellyroll']['length'],
        "area": calculated_data['jellyroll']['area'],
        "comp_thickness_p": calculated_data['p']['comp_thickness'] * 10000,
        "comp_thickness_n": calculated_data['n']['comp_thickness'] * 10000,
        "np_ratio": np_ratio,
        "Average Voltage (V)": voltage,
        "Specific capacity (Ah/kg)": (calculated_data['cell']['capacity']
                                      / calculated_data['cell']['mass']*1000),
        "Volumetric capacity (Ah/L)": (calculated_data['cell']['energy_vol']
                                       / voltage),
        "Name": "Calculated cell nr " + str(st.session_state.df_state + 1),
        "Form factor": "Cylindrical " + celltype,
        "Capacity calculation method": "Cell",
        "Technology": "Battery",
        "Category": "Calculated",
        "Cycle life": 1,
        "Maturity": "Commercial, Development, Research"
    }
    return results


def print_results(results):

    st.subheader("Results")
    st.write(
        f"**Cell capacity:** {results['Capacity (Ah)']:.2f} Ah | "
        f"{results['mass']:.1f} g"
    )

    st.write(
        f"**Cell energy:** {results['Energy (Wh)']:.1f} Wh | "
        f"{results['Specific Energy (Wh/kg)']:.1f} Wh/kg | "
        f"{results['Energy density (Wh/L)']:.1f} Wh/L"
    )

    st.write(
        f"**First cycle capacity loss:** "
        f"{results['firstcycleloss'] * 100:.1f}%"
    )

    st.write(
        f"**Estimated cell resistance:** "
        f"{results['Internal resistance (mOhm)']:.2f} mΩ"
    )

    st.write(
        f"**Jellyroll:** turns {results['turns']:.1f} "
        f"length: {results['length']:.2f} cm;"
        f"total electrode area: {2 * results['area']:.2f} cm^2"
    )

    st.write(
        f"+ve electrode: {results['comp_thickness_p']:.1f} µm thick | "
        f"-ve electrode: {results['comp_thickness_n']:.1f} µm | "
        f"n/p ratio: {results['np_ratio']:.1f}"
    )


def ui():

    # Define UI
    st.title("Estimating energy density of batteries")
    st.subheader("About")
    st.markdown(
        """
        This calculator estimates cell-level energy density for alkali-ion 
        or alkali-metal batteries (e.g. Li-ion, Li metal, Na-ion, Mg-ion)
        based on specifications of the constituent parts. This calculator 
        uses either the 18650 or 21700 form factors as the basis for the
        calculations. The default values for the positive and negative 
        electrodes are based on the suggested baseline data for NMC532 
        vs graphite presented by 
        *Harlow et al, J. Electrochem. Soc. 166 (13) A3031-A3044 (2019)*.  
        """
    )
    st.markdown(
        """
        Author: *Matt Lacey*, adapted to Wattrank by *Marcin Orzech*.  
        This calculator was originally developed by Matt Lacey 
        and published on lacey.se.
        """
    )
    st.info(
        """Adjust the values in the boxes below and press 'Calculate'
        to see the results. The results of the calculations are
        automatically added to all plots with 'Calculated' category."""
    )
    st.write("---")


def run_calc():
    ui()
    # UI elements for input
    st.subheader("Cell type:")
    celltype = st.radio("type", ("18650", "21700"),
                        label_visibility="collapsed")
    if st.button("Reset values to default"):
        st.session_state["calc"] += 1
    user_inputs = input_fields()

    st.write("---")

    calculate_button = st.button("Calculate❕", type="primary")
    if calculate_button:
        calculated_data = cyl_calculate(**user_inputs, **cell_param(celltype))
        results = formatted_results(calculated_data, celltype)
        c1, c2 = st.columns(2)
        with c1:
            print_results(results)
        with c2:
            st.plotly_chart(plot_cross_section(calculated_data), use_container_width=True)
        st.plotly_chart(plot_mass(calculated_data), use_container_width=True)

        return results


# if __name__ == "__main__":
#     run_calc()
