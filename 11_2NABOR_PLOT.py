import os
import plotly.io as pio
pio.renderers.default = "browser"
import datetime
from xlrd.xldate import xldate_as_datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Сброс ограничений на количество выводимых рядов
pd.set_option('display.max_rows', None)
# Сброс ограничений на число столбцов
pd.set_option('display.max_columns', None)
# Сброс ограничений на количество символов в записи
pd.set_option('display.max_colwidth', None)


def conditions(row):
    return (
            (
                    (row['Расход на входе(л/с)'] < 25)  # Расход на входе меньше 20 = не работает хотя бы один насос
            #  or ((row['Ходы насоса(ход/мин)'] == 0) or (
            #                     row['Ходы насоса(ход/мин).1'] == 0)))  # Ход насоса нулевой хотя бы по одному из насосов
            #
            # and (row['WithinInterval'] == 1
            )
    )

df = pd.read_csv('data/07/prep_data_target_11.csv')

# Apply conditions to create the target column
df['rashod_do_20'] = df.apply(conditions, axis=1).astype(int)

fig = go.Figure()

# Create figure with subplots
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08)
# Add original time series traces to each subplot
for i in range(3):
    # Add new y-axis for target

    if i == 0:
        for column in ['Ходы насоса(ход/мин)', 'Ходы насоса(ход/мин).1' , 'Расход на входе(л/с)']:
            fig.add_trace(go.Scatter(x=df['datetime'], y=df[column], mode='lines', name=column), row=i + 1, col=1)
    elif i == 1:
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['WithinInterval'], mode='lines', name='НПВ'), row=i + 1, col=1)
    elif i == 2:
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['rashod_do_20'], mode='lines', name='Расход на входе до 25'), row=i + 1, col=1)


fig.update_layout(
    title=f'Графический анализ расхождений фактических данных об НПВ и данных по расходу на входе',
    showlegend=True,
    # height=900
)

fig.show()

fig.write_html(r"C:\Users\olesya.krasnukhina\PycharmProjects\NSH\11_2NABOR_PLOT.html")