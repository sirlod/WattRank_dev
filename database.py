# -*- coding: utf-8 -*-
"""
Created on Sun May  1 13:21:11 2022.

@author: Marcin Orzech
"""

import os
import psycopg2
import pandas as pd

# Fetch env variables
USER = os.environ.get("user")
PASSWORD = os.environ.get("password")
HOST = os.environ.get("host")
PORT = os.environ.get("port")
DBNAME = os.environ.get("dbname")

columns_in_order = """
                    name,
                    specific_energy,
                    specific_power,
                    specific_power_peak,
                    energy_density,
                    ocv,
                    low_volt,
                    up_volt,
                    c_rate_dis,
                    c_rate_ch,
                    resistance,
                    specific_capacity,
                    volumetric_capacity,
                    capacity,
                    energy,
                    capacity_calc_method,
                    technology,
                    category,
                    cathode,
                    anode,
                    electrolyte,
                    form_factor,
                    cycle_life,
                    temperatur,
                    tags,
                    publication_date,
                    maturity,
                    reference"""
# DATA_TABLE = "SELECT * FROM data ORDER BY id;"
DATA_TABLE = f"""SELECT id, {columns_in_order} FROM data ORDER BY id;"""
PARAMETERS_TABLE = "SELECT * FROM parameters;"
INSERT_EMAIL = "INSERT INTO emails (cell_id, email) VALUES ((SELECT MAX(id) from data), %s)"
RESET_SEQUENCE = "SELECT setval('data_id_seq', (SELECT MAX(id) from data));"

def get_connection():
    return psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME,
        sslmode='require'
    )

def get_table(table):
    connection = get_connection()
    try:
        if table == 'data':
            return pd.read_sql(DATA_TABLE, connection, index_col='id')
        elif table == 'parameters':
            return pd.read_sql(PARAMETERS_TABLE, connection, index_col=None)
    finally:
        connection.close()


def upload_row(values):
    vals_str = ','.join(['%s' for i in range(len(values))])
    UPLOAD_DATA = f"INSERT INTO data ({columns_in_order}) VALUES ({vals_str});"
    connection = get_connection()
    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(RESET_SEQUENCE)
                cursor.execute(UPLOAD_DATA, values)
    finally:
        connection.close()


def save_email(address):
    connection = get_connection()
    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(INSERT_EMAIL, (address,))
    finally:
        connection.close()
