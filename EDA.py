from zipfile import ZipFile

# from pyforest import *
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
import datetime
from xlrd.xldate import xldate_as_datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Сброс ограничений на количество выводимых рядов
pd.set_option('display.max_rows', None)
# Сброс ограничений на число столбцов
pd.set_option('display.max_columns', None)
# Сброс ограничений на количество символов в записи
pd.set_option('display.max_colwidth', None)

########################################################################################################################
# looping opening and concating dataframes to construct full csv for each skvazhina
data = pd.DataFrame()
n = 0
with ZipFile('data/rig_11_451.zip') as myzip:
    namelist = myzip.namelist()
    for name in namelist:
        if name in namelist:
            print(f'Open {name}')
            myzip.extract(name)
            data_name = pd.read_excel(f'{name}', header=0,
                                      # usecols='A, AB',
                                      converters={
                                      'datetime': lambda x: xldate_as_datetime(float(x), 0)
                                      }
                                     )
            data_name = data_name.set_index('datetime', inplace=True)
            data_name = data_name.resample('T').mean()
            data_name = data_name.reset_index(inplace=True)
            data = pd.concat([data, pd.DataFrame(data_name)])
            n += 1

data.to_csv('data/data_agg_per_min_mean_rig_11_451.csv', index=False)
df = pd.read_csv('data/data_11_451_full.csv')
df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')


list_of_zip = ['rig_11_452.zip', 'rig_7.zip']
for zip_name in list_of_zip:
    # looping opening and concating dataframes
    data = pd.DataFrame()
    n = 0
    with ZipFile(f'data/{zip_name}') as myzip:
        namelist = myzip.namelist()
        for name in namelist:
            if name in namelist:
                print(f'Open {name}')
                myzip.extract(name)
                data_name = pd.read_excel(f'{name}', header=0,
                                          # usecols='A, AB',
                                          converters={
                                          'datetime': lambda x: xldate_as_datetime(float(x), 0)
                                          }
                                         )
                # data_name = data_name.set_index('datetime', inplace=True)
                # data_name = data_name.resample('T').mean()
                # data_name = data_name.reset_index(inplace=True)
                data = pd.concat([data, pd.DataFrame(data_name)])
                n += 1
    zip_name = zip_name.replace('.zip', '')
    data.to_csv(f'data/data_{zip_name}_full.csv', index=False)
    # df = pd.read_csv('data/data_11_451_full.csv')
    # df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')





list_of_zip = ['rig_11_451.zip', 'rig_11_452.zip', 'rig_7.zip']
for zip_name in list_of_zip:
    # looping opening and concating dataframes
    data = pd.DataFrame()
    n = 0
    with ZipFile(f'data/{zip_name}') as myzip:
        namelist = myzip.namelist()
        for name in namelist:
            if name in namelist:
                print(f'Open {name}')
                myzip.extract(name)
                data_name = pd.read_excel(f'{name}', header=0,
                                          usecols='A, AB',
                                          converters={
                                          'datetime': lambda x: xldate_as_datetime(float(x), 0)
                                          }
                                         )
                data_name['datetime'] = pd.to_datetime(data_name['datetime'], format='%Y-%m-%d %H:%M:%S')
                data_name = data_name.set_index('datetime', inplace=True)
                data_name = data_name.resample('T').mean()
                data_name = data_name.reset_index(inplace=True)
                data = pd.concat([data, pd.DataFrame(data_name)])
                n += 1
    zip_name = zip_name.replace('.zip', '')
    # data.to_csv(f'data/data_{zip_name}_full.csv', index=False)


######################################one graph with key features#######################################################

data = pd.read_csv('data/data_11_451_full.csv')
data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d %H:%M:%S')
data.set_index('datetime', inplace=True)
data_new = data.resample('T').mean()
data.reset_index(inplace=True)

describe_df = data_new.describe()
nunique_df = data_new.nunique()

data_new = data_new[nunique_df[(nunique_df != 0) & (nunique_df !=1)].index]

# Correlation matrix
correlation_matrix = data_new.corr()

# Heatmap for correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

# data_new_interval = data_new.loc['24.11.2023  1:00:00':'24.11.2023  8:00:00']
# data_new_interval = data_new.loc['14.12.2023  0:00:00':'16.12.2023  0:00:00']
data_new_interval = data_new.copy()

# Define datetime intervals for vertical lines
vertical_lines = [
    pd.Timestamp(pd.to_datetime('24.11.2023  4:00:00', format='%d.%m.%Y %H:%M:%S')), pd.Timestamp(pd.to_datetime('24.11.2023  5:00:00', format='%d.%m.%Y %H:%M:%S')),
    pd.Timestamp(pd.to_datetime('03.12.2023  10:30:00', format='%d.%m.%Y %H:%M:%S')), pd.Timestamp(pd.to_datetime('03.12.2023  11:30:00', format='%d.%m.%Y %H:%M:%S')),
    pd.Timestamp(pd.to_datetime('04.12.2023  6:10:00', format='%d.%m.%Y %H:%M:%S')), pd.Timestamp(pd.to_datetime('04.12.2023  7:10:00', format='%d.%m.%Y %H:%M:%S')),
    pd.Timestamp(pd.to_datetime('04.12.2023  8:00:00', format='%d.%m.%Y %H:%M:%S')), pd.Timestamp(pd.to_datetime('09.12.2023  0:00:00', format='%d.%m.%Y %H:%M:%S')),
    pd.Timestamp(pd.to_datetime('09.12.2023  19:40:00', format='%d.%m.%Y %H:%M:%S')), pd.Timestamp(pd.to_datetime('09.12.2023  20:50:00', format='%d.%m.%Y %H:%M:%S')),
    pd.Timestamp(pd.to_datetime('10.12.2023  10:30:00', format='%d.%m.%Y %H:%M:%S')), pd.Timestamp(pd.to_datetime('10.12.2023  15:30:00', format='%d.%m.%Y %H:%M:%S')),
    pd.Timestamp(pd.to_datetime('14.12.2023  5:50:00', format='%d.%m.%Y %H:%M:%S')), pd.Timestamp(pd.to_datetime('14.12.2023  7:10:00', format='%d.%m.%Y %H:%M:%S'))
                  ]

ts = pd.Series(data_new_interval['Расход на входе(л/с)'], index=data_new_interval.index)
ts1 = pd.Series(data_new_interval['Ходы насоса(ход/мин).1'], index=data_new_interval.index)
ts2 = pd.Series(data_new_interval['Ходы насоса(ход/мин)'], index=data_new_interval.index)
ts3 = pd.Series(data_new_interval['Нагрузка на долото(тс)'], index=data_new_interval.index)
ts4 = pd.Series(data_new_interval['Вес на крюке(тс)'], index=data_new_interval.index)
# Create figure
fig = go.Figure()

# Add original time series trace
fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines', name='Расход на входе(л/с)'))

# # Add rolling average trace
fig.add_trace(go.Scatter(x=ts1.index, y=ts1.values, mode='lines', name='Ходы насоса(ход/мин).1'))

# # Add rolling average trace
fig.add_trace(go.Scatter(x=ts2.index, y=ts3.values, mode='lines', name='Ходы насоса(ход/мин)'))

# # Add rolling average trace
fig.add_trace(go.Scatter(x=ts3.index, y=ts4.values, mode='lines', name='Нагрузка на долото(тс)'))

# # Add rolling average trace
fig.add_trace(go.Scatter(x=ts4.index, y=ts4.values, mode='lines', name='Вес на крюке(тс)'))

# Define zones for background coloring
y_zones = [0, 20, 40, np.inf]
zone_colors = ['rgba(255, 255, 255, 0)', 'rgba(200, 200, 255, 0.5)', 'rgba(255, 200, 200, 0.5)']

# Add shapes for background zones
for i in range(len(y_zones) - 1):
    fig.add_shape(
        type="rect",
        x0=ts.index[0],
        y0=y_zones[i],
        x1=ts.index[-1],
        y1=y_zones[i+1],
        fillcolor=zone_colors[i],
        layer="below",
        line=dict(color="rgba(0, 0, 0, 0)")
    )

# Add horizontal lines at 20 and 40
for y in [20, 40]:
    fig.add_shape(type="line",
        x0=ts.index[0], y0=y,
        x1=ts.index[-1], y1=y,
        line=dict(color="rgba(0,0,0,0.5)", width=1, dash="dash"),
    )

# Add vertical lines
for line in vertical_lines:
    fig.add_shape(type="line",
        x0=line, y0=ts.min(),
        x1=line, y1=ts.max(),
        line=dict(color="rgba(255,0,0,0.5)", width=1, dash="dash"),
    )

# Update layout
fig.update_layout(
    title='Данные по бурению скважин',
    xaxis_title='Date',
    yaxis_title='Value',
    showlegend=True
)

# Show plot
fig.show()

########################################################################################################################
# BURENIE
bur_list = ['Скорость проходки(м/ч)',
            'Глубина забоя(м)',
            'Нагрузка на долото(тс)',
            'Обороты СВП(об/мин)',
            'Глубина инструмента(м)',
            'Уровень(м3)',
            'Уровень(м3).1',
            'Уровень(м3).2',
            'Уровень(м3).3',
            'Момент на роторе(кНм)',
            'Момент на СВП(кНм)',
            'Вес на крюке(тс)',
            'Положение крюкоблока(м)']
#NARASHIVANIE
nar_list = ['Уровень(м3)',
            'Уровень(м3).1',
            'Уровень(м3).2',
            'Уровень(м3).3',
            'Глубина инструмента(м)',
            'Скорость проходки(м/ч)',
            'Момент на маш.ключе(кНм)',
            'Момент на маш.ключе(кНм).1',
            'Скорость СПО(м/с)',
            'Момент на ключе(кНм)',
            'Наработка каната(т*км)'
            'Вес на крюке(тс)',
            'Положение крюкоблока(м)']

list_of_path = ['data/data_11_451_full.csv', 'data/data_rig_11_452_full.csv', 'data/data_rig_7_full.csv']

i = 0
path = list_of_path[0]
data = pd.read_csv(path)
data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d %H:%M:%S')
data.set_index('datetime', inplace=True)
data_new = data.resample('T').mean()
data.reset_index(inplace=True)

describe_df = data_new.describe()
nunique_df = data_new.nunique()

data_new = data_new[nunique_df[(nunique_df != 0) & (nunique_df !=1)].index]

# Correlation matrix
correlation_matrix = data_new.corr()

# Heatmap for correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title(f'Correlation Matrix for {path}')
plt.show()

# data_new_interval = data_new.loc['24.11.2023  1:00:00':'24.11.2023  8:00:00']
# data_new_interval = data_new.loc['14.12.2023  0:00:00':'16.12.2023  0:00:00']
data_new_interval = data_new.copy()

try:
    data_new.rename(columns={
        'Момент на ключе ZQ/ГКШ(кН*м)': 'Момент на ключе(кН*м)',
        'Ур.1 долив.(м3)': 'Уровень(м3)',
        'Ур.2 воронка(м3)': 'Уровень(м3).1',
        'Ур.3 рабоч.2(м3)': 'Уровень(м3).3',
        'Ур.4 рабоч.1(м3)': 'Уровень(м3).2',
        'Момент на маш.ключе(TQ)': 'Момент на маш.ключе(кН*м).1',
        'Температура окр. среды(C)': 'Температура окр.среды(C)'
    }, inplace=True)
except Exception:
    print('Ошибка при переименовании столбцов')

# data_new_interval = data_new.loc['14.12.2023  0:00:00':'16.12.2023  0:00:00']
data_new_interval = data_new.copy()
if path.__contains__('451'):
    # Define datetime intervals for vertical lines
    vertical_lines = [
        pd.Timestamp(pd.to_datetime('24.11.2023  4:00:00', format='%d.%m.%Y %H:%M:%S')),
            pd.Timestamp(pd.to_datetime('24.11.2023  5:00:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('03.12.2023  10:30:00', format='%d.%m.%Y %H:%M:%S')),
            pd.Timestamp(pd.to_datetime('03.12.2023  11:30:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('04.12.2023  6:10:00', format='%d.%m.%Y %H:%M:%S')),
            pd.Timestamp(pd.to_datetime('04.12.2023  7:10:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('04.12.2023  8:00:00', format='%d.%m.%Y %H:%M:%S')),
            pd.Timestamp(pd.to_datetime('09.12.2023  0:00:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('09.12.2023  19:40:00', format='%d.%m.%Y %H:%M:%S')),
            pd.Timestamp(pd.to_datetime('09.12.2023  20:50:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('10.12.2023  10:30:00', format='%d.%m.%Y %H:%M:%S')),
            pd.Timestamp(pd.to_datetime('10.12.2023  15:30:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('14.12.2023  5:50:00', format='%d.%m.%Y %H:%M:%S')),
            pd.Timestamp(pd.to_datetime('14.12.2023  7:10:00', format='%d.%m.%Y %H:%M:%S'))
                      ] # for 451 data
elif path.__contains__('452'):
    vertical_lines = [
        pd.Timestamp(pd.to_datetime('09.01.2024 13:30:00', format='%d.%m.%Y %H:%M:%S')),
            pd.Timestamp(pd.to_datetime('09.01.2024 14:30:00', format='%d.%m.%Y %H:%M:%S'))
                      ] # for 452 data
elif path.__contains__('7'):
    vertical_lines = [
        pd.Timestamp(pd.to_datetime('08.12.2023 15:00:00', format='%d.%m.%Y %H:%M:%S')),
            pd.Timestamp(pd.to_datetime('08.12.2023 17:00:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('10.12.2023 10:45:00', format='%d.%m.%Y %H:%M:%S')),
            pd.Timestamp(pd.to_datetime('10.12.2023 12:00:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('11.12.2023 22:00:00', format='%d.%m.%Y %H:%M:%S')),
            pd.Timestamp(pd.to_datetime('11.12.2023 22:30:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('15.12.2023 0:30:00', format='%d.%m.%Y %H:%M:%S')),
            pd.Timestamp(pd.to_datetime('15.12.2023 1:15:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('17.12.2023 13:40:00', format='%d.%m.%Y %H:%M:%S')),
            pd.Timestamp(pd.to_datetime('17.12.2023 14:10:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('18.12.2023 13:45:00', format='%d.%m.%Y %H:%M:%S')),
            pd.Timestamp(pd.to_datetime('18.12.2023 21:30:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('19.12.2023 2:30:00', format='%d.%m.%Y %H:%M:%S')),
            pd.Timestamp(pd.to_datetime('19.12.2023 6:00:00', format='%d.%m.%Y %H:%M:%S'))
                      ] # for 174 data
else:
    print('Неизвестный путь')
    exit()



# PLOT DATA
# Create figure
fig = go.Figure()

# Define zones for background coloring
y_zones = [0, 20, 40, np.inf]
zone_colors = ['rgba(255, 255, 255, 0)', 'rgba(200, 200, 255, 0.5)', 'rgba(255, 200, 200, 0.5)']
# Create figure with subplots
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08)
# Add original time series traces to each subplot
for i in range(3):
    if i == 0:
        ts = pd.Series(data_new_interval['Скорость проходки(м/ч)'],
                       index=data_new_interval.index)
        ts1 = pd.Series(data_new_interval['Глубина забоя(м)'],
                        index=data_new_interval.index)
        ts2 = pd.Series(data_new_interval['Нагрузка на долото(тс)'],
                        index=data_new_interval.index)
        ts3 = pd.Series(data_new_interval['Обороты СВП(об/мин)'],
                        index=data_new_interval.index)
        ts4 = pd.Series(data_new_interval['Глубина инструмента(м)'],
                        index=data_new_interval.index)
        # ts9 = pd.Series(data_new_interval['Момент на роторе(кНм)'],
        #                 index=data_new_interval.index)
        ts10 = pd.Series(data_new_interval['Расход на входе(л/с)'],
                         index=data_new_interval.index)
        ts11 = pd.Series(data_new_interval['Момент на СВП(кН*м)'],
                         index=data_new_interval.index)
        ts12 = pd.Series(data_new_interval['Положение крюкоблока(м)'],
                         index=data_new_interval.index)
        ts13 = pd.Series(data_new_interval['Вес на крюке(тс)'],
                         index=data_new_interval.index)

        fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines',
                                 name='Скорость проходки(м/ч)'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=ts1.index, y=ts1.values, mode='lines',
                                 name='Глубина забоя(м)'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=ts2.index, y=ts2.values, mode='lines',
                                 name='Нагрузка на долото(тс)'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=ts3.index, y=ts3.values, mode='lines',
                                 name='Обороты СВП(об/мин)'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=ts4.index, y=ts4.values, mode='lines',
                                 name='Глубина инструмента(м)'), row=i+1, col=1)
        # fig.add_trace(go.Scatter(x=ts9.index, y=ts9.values, mode='lines',
        #                          name='Момент на роторе(кНм)'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=ts10.index, y=ts10.values, mode='lines',
                                 name='Расход на входе(л/с)'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=ts11.index, y=ts11.values, mode='lines',
                                 name='Момент на СВП(кН*м)'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=ts12.index, y=ts12.values, mode='lines',
                                 name='Положение крюкоблока(м)'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=ts13.index, y=ts13.values, mode='lines',
                                 name='Вес на крюке(тс)'), row=i+1, col=1)
    elif i == 1:

        # ts = pd.Series(data_new_interval['Скорость проходки(м/ч)'],
        # index=data_new_interval.index)
        ts1 = pd.Series(data_new_interval['Момент на ключе(кН*м)'],
                        index=data_new_interval.index)
        ts2 = pd.Series(data_new_interval['Нагрузка на долото(тс)'],
                        index=data_new_interval.index)
        ts3 = pd.Series(data_new_interval['Наработка каната(т*км)'],
                        index=data_new_interval.index)
        # ts4 = pd.Series(data_new_interval['Глубина инструмента(м)'],
        # index=data_new_interval.index)
        ts5 = pd.Series(data_new_interval['Уровень(м3)'],
                        index=data_new_interval.index)
        ts6 = pd.Series(data_new_interval['Уровень(м3).1'],
                        index=data_new_interval.index)
        ts7 = pd.Series(data_new_interval['Уровень(м3).2'],
                        index=data_new_interval.index)
        ts8 = pd.Series(data_new_interval['Уровень(м3).3'],
                        index=data_new_interval.index)
        ts9 = pd.Series(data_new_interval['Момент на маш.ключе(кН*м)'],
                        index=data_new_interval.index)
        ts10 = pd.Series(data_new_interval['Момент на маш.ключе(кН*м).1'],
                         index=data_new_interval.index)
        ts11 = pd.Series(data_new_interval['Скорость СПО(м/с)'],
                         index=data_new_interval.index)
        # ts12 = pd.Series(data_new_interval['Положение крюкоблока(м)'],
        # index=data_new_interval.index)
        # ts13 = pd.Series(data_new_interval['Вес на крюке(тс)'],
        # index=data_new_interval.index)

        # fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines',
        #                          name='Скорость проходки(м/ч)'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=ts1.index, y=ts1.values, mode='lines',
                                 name='Момент на ключе(кН*м)'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=ts2.index, y=ts2.values, mode='lines',
                                 name='Нагрузка на долото(тс)'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=ts3.index, y=ts3.values, mode='lines',
                                 name='Наработка каната(т*км)'), row=i+1, col=1)
        # fig.add_trace(go.Scatter(x=ts4.index, y=ts4.values, mode='lines',
        # name='Глубина инструмента(м)'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=ts5.index, y=ts5.values, mode='lines',
                                 name='Уровень(м3)'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=ts6.index, y=ts6.values, mode='lines',
                                 name='Уровень(м3).1'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=ts7.index, y=ts7.values, mode='lines',
                                 name='Уровень(м3).2'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=ts8.index, y=ts8.values, mode='lines',
                                 name='Уровень(м3).3'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=ts9.index, y=ts9.values, mode='lines',
                                 name='Момент на маш.ключе(кН*м)'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=ts10.index, y=ts10.values, mode='lines',
                                 name='Момент на маш.ключе(кН*м).1'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=ts11.index, y=ts11.values, mode='lines',
                                 name='Скорость СПО(м/с)'), row=i+1, col=1)
        # fig.add_trace(go.Scatter(x=ts12.index, y=ts12.values, mode='lines',
        # name='Положение крюкоблока(м)'), row=i+1, col=1)
        # fig.add_trace(go.Scatter(x=ts13.index, y=ts13.values, mode='lines',
        # name='Вес на крюке(тс)'), row=i+1, col=1)
    elif i == 2:

        ts = pd.Series(data_new_interval['Расход на входе(л/с)'],
                       index=data_new_interval.index)
        ts1 = pd.Series(data_new_interval['Ходы насоса(ход/мин)'],
                        index=data_new_interval.index)
        ts2 = pd.Series(data_new_interval['Ходы насоса(ход/мин).1'],
                        index=data_new_interval.index)
        ts3 = pd.Series(data_new_interval['Давление в манифольде(МПа)'],
                        index=data_new_interval.index)
        ts4 = pd.Series(data_new_interval['Температура окр.среды(C)'],
                        index=data_new_interval.index)

        fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines',
                                 name='Расход на входе(л/с)'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=ts1.index, y=ts1.values, mode='lines',
                                 name='Ходы насоса(ход/мин)'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=ts2.index, y=ts2.values, mode='lines',
                                 name='Ходы насоса(ход/мин).1'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=ts3.index, y=ts3.values, mode='lines',
                                 name='Давление в манифольде(МПа)'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=ts4.index, y=ts4.values, mode='lines',
                                 name='Температура окр.среды(C)'), row=i+1, col=1)

for i in range(3):
    # Add shapes for background zones
    for j in range(len(y_zones) - 1):
        fig.add_shape(
            type="rect",
            x0=ts.index[0],
            y0=y_zones[j],
            x1=ts.index[-1],
            y1=y_zones[j+1],
            fillcolor=zone_colors[j],
            layer="below",
            line=dict(color="rgba(0, 0, 0, 0)")
        )
    # Add horizontal lines at 20 and 40
    for y in [20, 40]:
        fig.add_shape(type="line",
            x0=ts.index[0], y0=y,
            x1=ts.index[-1], y1=y,
            line=dict(color="rgba(0,0,0,0.5)", width=1, dash="dash"),
        )
    # Add vertical lines
    for line in vertical_lines:
        fig.add_shape(type="line",
            x0=line, y0=0,
            x1=line, y1=1000,
            line=dict(color="rgba(255,0,0,0.5)", width=1, dash="dash"),
        )

        # Add vertical lines for target indices
    for index in target_indices:
        fig.add_vline(x=index, line=dict(color="rgba(255,0,0,0.5)", width=1, dash="dash"), row=i + 1, col=1)

# Update layout for each subplot
# for i in range(3):
#     fig.update_xaxes(title_text="Date", row=i+1, col=1)
#     fig.update_yaxes(title_text="Value", row=i+1, col=1)
# Update layout for the whole figure
fig.update_layout(
    title= f'Показатели бурения нефтяных скважин для {path}',
    showlegend=True,
    # height=900
)
# and legends for each subplot
# fig.update_annotations({'xref': 'paper', 'yref': 'paper', 'x': 0.5, 'y': 1.15, 'showarrow': False, 'text': 'Plot 1'}, row=1, col=1)
# fig.update_annotations({'xref': 'paper', 'yref': 'paper', 'x': 0.5, 'y': 0.78, 'showarrow': False, 'text': 'Plot 2'}, row=2, col=1)
# fig.update_annotations({'xref': 'paper', 'yref': 'paper', 'x': 0.5, 'y': 0.45, 'showarrow': False, 'text': 'Plot 3'}, row=3, col=1)
# Show plot
fig.show()



'''
Критерии выставления единиц
Исключение синего и желтого кругов:

Расход на входе меньше 20 = не работает хотя бы один насос (из ответов на вопросы на почте от заказчика и обсуждения с Мишей)
Ход насоса нулевой хотя бы по одному из насосов (из первичной постановки задачи)
Исключение коричневого круга:

Вес на крюке не падает и не нулевой (дельта по минутам отрицательная или нулевая) (из обсуждения с Мишей)
Исключение остановки по погодным условиям:

Температура выше -40 (из ответов на вопросы на почте от заказчика)
Алгоритм
Итерируясь по интервалам из документа Ремонт_БН_Гидроблок:

Вручную расширить\сузить\уточнить исследуемый интервал, к которому будут применять критерии, описанные выше

По выделенному интервалу прогнать скрипт выделяющий наблюдения по критериям, описанным выше

Опираясь на минутные данные, проставить единицы в посекундных данных датафрейма при выполнении всех критериев, описанных выше
'''

data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d %H:%M:%S')
data = data.set_index('datetime', inplace=True)

data_new = data_new.sort_index()
data_new_interval_test = data_new.loc['2023-12-08 07:16:00':'2023-12-09 00:29:00']

data = data.sort_index()
data_sec_test = data.loc['2023-12-08 07:16:00':'2023-12-09 00:29:00']

data_new_interval_test['delta_weight'] = data_new_interval_test['Вес на крюке(тс)'].diff()

# Define your conditions
def conditions(row):
    return (
            (row['Расход на входе(л/с)'] < 25) # Расход на входе меньше 20 = не работает хотя бы один насос
            and ((row['Ходы насоса(ход/мин)'] == 0) or (row['Ходы насоса(ход/мин).1'] == 0)) # Ход насоса нулевой хотя бы по одному из насосов
            and ((row['delta_weight'] == 0) or (row['delta_weight'] < 0))
    )  # Example conditions


# Apply conditions to create the target column
data_new_interval_test['target'] = data_new_interval_test.apply(conditions, axis=1).astype(int)

# Fill the target column in second-level data with minute-level data
data_sec_test['target'] = data_sec_test.index.map(data_new_interval_test['target'])

# Forward fill NaN values in case of any missing minutes
df['target'] = df['target'].ffill()


data = pd.read_csv('data/data_rig_7_full.csv')

list_of_path = ['data/data_11_451_full.csv', 'data/data_rig_11_452_full.csv', 'data/data_rig_7_full.csv']

i = 2
path = list_of_path[0]
data_new_interval = data_new_interval_test.copy()

try:
    data_new.rename(columns={
        'Момент на ключе ZQ/ГКШ(кН*м)': 'Момент на ключе(кН*м)',
        'Ур.1 долив.(м3)': 'Уровень(м3)',
        'Ур.2 воронка(м3)': 'Уровень(м3).1',
        'Ур.3 рабоч.2(м3)': 'Уровень(м3).3',
        'Ур.4 рабоч.1(м3)': 'Уровень(м3).2',
        'Момент на маш.ключе(TQ)': 'Момент на маш.ключе(кН*м).1',
        'Температура окр. среды(C)': 'Температура окр.среды(C)'
    }, inplace=True)
except Exception:
    print('Ошибка при переименовании столбцов')

if path.__contains__('451'):
    # Define datetime intervals for vertical lines
    vertical_lines = [
        pd.Timestamp(pd.to_datetime('24.11.2023  4:00:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('24.11.2023  5:00:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('03.12.2023  10:30:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('03.12.2023  11:30:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('04.12.2023  6:10:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('04.12.2023  7:10:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('04.12.2023  8:00:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('09.12.2023  0:00:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('09.12.2023  19:40:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('09.12.2023  20:50:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('10.12.2023  10:30:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('10.12.2023  15:30:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('14.12.2023  5:50:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('14.12.2023  7:10:00', format='%d.%m.%Y %H:%M:%S'))
    ]  # for 451 data
elif path.__contains__('452'):
    vertical_lines = [
        pd.Timestamp(pd.to_datetime('09.01.2024 13:30:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('09.01.2024 14:30:00', format='%d.%m.%Y %H:%M:%S'))
    ]  # for 452 data
elif path.__contains__('7'):
    vertical_lines = [
        pd.Timestamp(pd.to_datetime('08.12.2023 15:00:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('08.12.2023 17:00:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('10.12.2023 10:45:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('10.12.2023 12:00:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('11.12.2023 22:00:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('11.12.2023 22:30:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('15.12.2023 0:30:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('15.12.2023 1:15:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('17.12.2023 13:40:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('17.12.2023 14:10:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('18.12.2023 13:45:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('18.12.2023 21:30:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('19.12.2023 2:30:00', format='%d.%m.%Y %H:%M:%S')),
        pd.Timestamp(pd.to_datetime('19.12.2023 6:00:00', format='%d.%m.%Y %H:%M:%S'))
    ]  # for 174 data
else:
    print('Неизвестный путь')
    exit()

# Generate sample target data (binary column with 0 or 1)
target = pd.Series(data_new_interval['target'],
                   index=data_new_interval.index)

# Find indices where target is 1
target_indices = target[target == 1].index

# PLOT DATA
# Create figure
fig = go.Figure()

# Define zones for background coloring
y_zones = [0, 20, 40, np.inf]
zone_colors = ['rgba(255, 255, 255, 0)', 'rgba(200, 200, 255, 0.5)', 'rgba(255, 200, 200, 0.5)']
# Create figure with subplots
fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.08)
# Add original time series traces to each subplot
for i in range(4):
    # Add new y-axis for target

    if i == 0:
        ts = pd.Series(data_new_interval['Скорость проходки(м/ч)'],
                       index=data_new_interval.index)
        ts1 = pd.Series(data_new_interval['Глубина забоя(м)'],
                        index=data_new_interval.index)
        ts2 = pd.Series(data_new_interval['Нагрузка на долото(тс)'],
                        index=data_new_interval.index)
        ts3 = pd.Series(data_new_interval['Обороты СВП(об/мин)'],
                        index=data_new_interval.index)
        ts4 = pd.Series(data_new_interval['Глубина инструмента(м)'],
                        index=data_new_interval.index)
        # ts9 = pd.Series(data_new_interval['Момент на роторе(кНм)'],
        #                 index=data_new_interval.index)
        ts10 = pd.Series(data_new_interval['Расход на входе(л/с)'],
                         index=data_new_interval.index)
        ts11 = pd.Series(data_new_interval['Момент на СВП(кН*м)'],
                         index=data_new_interval.index)
        ts12 = pd.Series(data_new_interval['Положение крюкоблока(м)'],
                         index=data_new_interval.index)
        ts13 = pd.Series(data_new_interval['Вес на крюке(тс)'],
                         index=data_new_interval.index)

        fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines',
                                 name='Скорость проходки(м/ч)'), row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=ts1.index, y=ts1.values, mode='lines',
                                 name='Глубина забоя(м)'), row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=ts2.index, y=ts2.values, mode='lines',
                                 name='Нагрузка на долото(тс)'), row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=ts3.index, y=ts3.values, mode='lines',
                                 name='Обороты СВП(об/мин)'), row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=ts4.index, y=ts4.values, mode='lines',
                                 name='Глубина инструмента(м)'), row=i + 1, col=1)
        # fig.add_trace(go.Scatter(x=ts9.index, y=ts9.values, mode='lines',
        #                          name='Момент на роторе(кНм)'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=ts10.index, y=ts10.values, mode='lines',
                                 name='Расход на входе(л/с)'), row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=ts11.index, y=ts11.values, mode='lines',
                                 name='Момент на СВП(кН*м)'), row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=ts12.index, y=ts12.values, mode='lines',
                                 name='Положение крюкоблока(м)'), row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=ts13.index, y=ts13.values, mode='lines',
                                 name='Вес на крюке(тс)'), row=i + 1, col=1)
    elif i == 1:

        # ts = pd.Series(data_new_interval['Скорость проходки(м/ч)'],
        # index=data_new_interval.index)
        ts1 = pd.Series(data_new_interval['Момент на ключе(кН*м)'],
                        index=data_new_interval.index)
        ts2 = pd.Series(data_new_interval['Нагрузка на долото(тс)'],
                        index=data_new_interval.index)
        ts3 = pd.Series(data_new_interval['Наработка каната(т*км)'],
                        index=data_new_interval.index)
        # ts4 = pd.Series(data_new_interval['Глубина инструмента(м)'],
        # index=data_new_interval.index)
        ts5 = pd.Series(data_new_interval['Уровень(м3)'],
                        index=data_new_interval.index)
        ts6 = pd.Series(data_new_interval['Уровень(м3).1'],
                        index=data_new_interval.index)
        ts7 = pd.Series(data_new_interval['Уровень(м3).2'],
                        index=data_new_interval.index)
        ts8 = pd.Series(data_new_interval['Уровень(м3).3'],
                        index=data_new_interval.index)
        ts9 = pd.Series(data_new_interval['Момент на маш.ключе(кН*м)'],
                        index=data_new_interval.index)
        ts10 = pd.Series(data_new_interval['Момент на маш.ключе(кН*м).1'],
                         index=data_new_interval.index)
        ts11 = pd.Series(data_new_interval['Скорость СПО(м/с)'],
                         index=data_new_interval.index)
        # ts12 = pd.Series(data_new_interval['Положение крюкоблока(м)'],
        # index=data_new_interval.index)
        # ts13 = pd.Series(data_new_interval['Вес на крюке(тс)'],
        # index=data_new_interval.index)

        # fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines',
        #                          name='Скорость проходки(м/ч)'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=ts1.index, y=ts1.values, mode='lines',
                                 name='Момент на ключе(кН*м)'), row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=ts2.index, y=ts2.values, mode='lines',
                                 name='Нагрузка на долото(тс)'), row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=ts3.index, y=ts3.values, mode='lines',
                                 name='Наработка каната(т*км)'), row=i + 1, col=1)
        # fig.add_trace(go.Scatter(x=ts4.index, y=ts4.values, mode='lines',
        # name='Глубина инструмента(м)'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=ts5.index, y=ts5.values, mode='lines',
                                 name='Уровень(м3)'), row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=ts6.index, y=ts6.values, mode='lines',
                                 name='Уровень(м3).1'), row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=ts7.index, y=ts7.values, mode='lines',
                                 name='Уровень(м3).2'), row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=ts8.index, y=ts8.values, mode='lines',
                                 name='Уровень(м3).3'), row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=ts9.index, y=ts9.values, mode='lines',
                                 name='Момент на маш.ключе(кН*м)'), row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=ts10.index, y=ts10.values, mode='lines',
                                 name='Момент на маш.ключе(кН*м).1'), row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=ts11.index, y=ts11.values, mode='lines',
                                 name='Скорость СПО(м/с)'), row=i + 1, col=1)
        # fig.add_trace(go.Scatter(x=ts12.index, y=ts12.values, mode='lines',
        # name='Положение крюкоблока(м)'), row=i+1, col=1)
        # fig.add_trace(go.Scatter(x=ts13.index, y=ts13.values, mode='lines',
        # name='Вес на крюке(тс)'), row=i+1, col=1)
    elif i == 2:

        ts = pd.Series(data_new_interval['Расход на входе(л/с)'],
                       index=data_new_interval.index)
        ts1 = pd.Series(data_new_interval['Ходы насоса(ход/мин)'],
                        index=data_new_interval.index)
        ts2 = pd.Series(data_new_interval['Ходы насоса(ход/мин).1'],
                        index=data_new_interval.index)
        ts3 = pd.Series(data_new_interval['Давление в манифольде(МПа)'],
                        index=data_new_interval.index)
        ts4 = pd.Series(data_new_interval['Температура окр.среды(C)'],
                        index=data_new_interval.index)

        fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines',
                                 name='Расход на входе(л/с)'), row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=ts1.index, y=ts1.values, mode='lines',
                                 name='Ходы насоса(ход/мин)'), row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=ts2.index, y=ts2.values, mode='lines',
                                 name='Ходы насоса(ход/мин).1'), row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=ts3.index, y=ts3.values, mode='lines',
                                 name='Давление в манифольде(МПа)'), row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=ts4.index, y=ts4.values, mode='lines',
                                 name='Температура окр.среды(C)'), row=i + 1, col=1)
    elif i == 3:
        fig.add_trace(go.Scatter(x=target.index, y=target.values, mode='lines', name='Target',
                                 line=dict(color='green')), row=i + 1, col=1)

for i in range(3):
    # Add shapes for background zones
    for j in range(len(y_zones) - 1):
        fig.add_shape(
            type="rect",
            x0=ts.index[0],
            y0=y_zones[j],
            x1=ts.index[-1],
            y1=y_zones[j + 1],
            fillcolor=zone_colors[j],
            layer="below",
            line=dict(color="rgba(0, 0, 0, 0)")
        )
    # Add horizontal lines at 20 and 40
    for y in [20, 40]:
        fig.add_shape(type="line",
                      x0=ts.index[0], y0=y,
                      x1=ts.index[-1], y1=y,
                      line=dict(color="rgba(0,0,0,0.5)", width=1, dash="dash"),
                      )
    # Add vertical lines
    for line in vertical_lines:
        fig.add_shape(type="line",
                      x0=line, y0=0,
                      x1=line, y1=1000,
                      line=dict(color="rgba(255,0,0,0.5)", width=1, dash="dash"),
                      )

# Update layout for each subplot
# for i in range(3):
#     fig.update_xaxes(title_text="Date", row=i+1, col=1)
#     fig.update_yaxes(title_text="Value", row=i+1, col=1)
# Update layout for each subplot
# for i in range(3):
#     fig.update_xaxes(title_text="Date", row=i+1, col=1)
#     fig.update_yaxes(title_text="Value", row=i+1, col=1)
# Update layout for the whole figure
fig.update_layout(
    title=f'Показатели бурения нефтяных скважин для {path}',
    showlegend=True,
    # height=900
)
# and legends for each subplot
# fig.update_annotations({'xref': 'paper', 'yref': 'paper', 'x': 0.5, 'y': 1.15, 'showarrow': False, 'text': 'Plot 1'}, row=1, col=1)
# fig.update_annotations({'xref': 'paper', 'yref': 'paper', 'x': 0.5, 'y': 0.78, 'showarrow': False, 'text': 'Plot 2'}, row=2, col=1)
# fig.update_annotations({'xref': 'paper', 'yref': 'paper', 'x': 0.5, 'y': 0.45, 'showarrow': False, 'text': 'Plot 3'}, row=3, col=1)
# Show plot
fig.show()