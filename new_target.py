import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
import datetime
from xlrd.xldate import xldate_as_datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import datetime

data_new_interval = pd.read_csv('data/prep_data_451_452.csv')
time_intervals = pd.read_csv('data/intervals.csv', delimiter=';')

intervals = [tuple(i) for i in time_intervals[(time_intervals['Column4'] == 'НПВ') & (time_intervals['Column5'] == 'Ремонт')][['Column2', 'Column3', 'Column6']].values.tolist()]


def create_binary_column_optimized(df, intervals):
    """
    Creates a binary column indicating whether the datetime index falls within any of the specified intervals.

    Parameters:
    df (DataFrame): Input DataFrame with a datetime index.
    intervals (list of tuples): List of tuples containing start and end datetime strings for each interval.

    Returns:
    DataFrame: DataFrame with a new binary column indicating whether the index falls within any of the intervals.
    """
    # Convert the datetime index to Timestamp
    df_index = pd.to_datetime(df.index)

    # Initialize an empty Series to store the binary values
    binary_target = pd.Series(0, index=df.index)

    # Iterate through each interval
    for start_dt, end_dt, type in intervals:
        # start_dt = pd.to_datetime(start)
        # end_dt = pd.to_datetime(end)

        # Update the binary values for rows within the current interval
        binary_target = binary_target | ((df_index >= start_dt) & (df_index <= end_dt))

    # Assign the binary values to the DataFrame
    df['binary_target'] = binary_target.astype(int)

    return df


# def create_column_with_type(df, intervals):
#     # Create a new column with default value as None
#     df['multi_target'] = None
#
#     # Iterate over each row
#     for index, row in df.iterrows():
#         # Get the datetime index
#         date_index = row.name
#
#         # Iterate over each tuple
#         for start_date, end_date, type in intervals:
#             # Check if the index falls between the start date and end date
#             start_date = pd.to_datetime(start_date)
#             end_date = pd.to_datetime(end_date)
#             if start_date <= date_index <= end_date:
#                 # Assign the type value to the 'New_Column'
#                 df.at[index, 'multi_target'] = type
#                 break  # No need to check other tuples once assigned
#         if not df.at[index, 'multi_target']:
#             df.at[index, 'multi_target'] = None
#
#     return df

# Define conditions
def conditions(row):
    return (
        ((row['Расход на входе(л/с)'] < 25) # Расход на входе меньше 20 = не работает хотя бы один насос
            or ((row['Ходы насоса(ход/мин)'] == 0) or (row['Ходы насоса(ход/мин).1'] == 0))) # Ход насоса нулевой хотя бы по одному из насосов
            # and ((row['delta_weight'] == 0) or (row['delta_weight'] < 0))
            # and ((row['Вес на крюке(тс)'] == 0) or (row['Вес на крюке(тс)'] < 0))
            and (row['WithinInterval'] == 1)
    )


# Binary column that indicates the intersection of index with npv intervals
data_new_interval['datetime'] = pd.to_datetime(data_new_interval['datetime'], format='%Y-%m-%d %H:%M:%S')
data_new_interval.set_index('datetime', inplace=True)
df_interpolated = create_binary_column_optimized(data_new_interval, intervals)
# df_interpolated = create_column_with_type(df_interpolated, intervals)

df_intervals = time_intervals[(time_intervals['Column4'] == 'НПВ') & (time_intervals['Column5'] == 'Ремонт')]
df_intervals['Column2'] = pd.to_datetime(df_intervals['Column2'], format='%Y-%m-%d %H:%M:%S')
df_intervals['Column3'] = pd.to_datetime(df_intervals['Column3'], format='%Y-%m-%d %H:%M:%S')
df_intervals = df_intervals.drop(['Column1', 'Column4', 'Column5', 'Column7'], axis = 1)

# Create a datetime index in minute frequency
datetime_index = pd.date_range(start=df_intervals['Column2'].min(), end=df_intervals['Column3'].max(), freq='T')
# Create an empty DataFrame with the datetime index
new_intervals = pd.DataFrame(index=datetime_index)
# Iterate over each row in the original DataFrame
i = 0
for index, row in df_intervals.iterrows():
    # Extract start date, end date, and type value
    start_date = row['Column2']
    end_date = row['Column3']
    type_value = row['Column6']
    # Filter the datetime index to include only those between start date and end date
    filtered_index = new_intervals.index[(new_intervals.index >= start_date) & (new_intervals.index <= end_date)]
    # Assign the type value to the filtered datetime index
    new_intervals.loc[filtered_index, 'type_value'] = type_value
    i += 1
    print(f'{i*100/len(df_intervals)} %')

new_intervals = new_intervals.dropna()

df_interpolated = df_interpolated.reset_index()
new_intervals = new_intervals.reset_index()
new_intervals = new_intervals.rename(columns = {'index' : 'datetime'})
merged_df = pd.merge(df_interpolated, new_intervals, on='datetime', how='left')

merged_df.set_index('datetime', inplace=True)

# Apply conditions to create the target column
# df_interpolated['target'] = df_interpolated.apply(conditions, axis=1).astype(int)
data_new_interval = merged_df.copy()

vertical_lines_start =  [pd.Timestamp(pd.to_datetime(i[0], format='%Y-%m-%d %H:%M:%S')) for i in intervals]
vertical_lines_end =  [pd.Timestamp(pd.to_datetime(i[1], format='%Y-%m-%d %H:%M:%S')) for i in intervals]


# Generate sample target data (binary column with 0 or 1)
binary_target = pd.Series(data_new_interval['binary_target'],
                    index=data_new_interval.index)

# Find indices where target is 1
binary_target_indices = binary_target[binary_target == 1].index

# Generate sample target data (binary column with 0 or 1)
multi_target = pd.Series(data_new_interval['multi_target'],
                    index=data_new_interval.index)

# Find indices where target is 1
multi_target_indices = multi_target[multi_target == 1].index

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
        # ts = pd.Series(data_new_interval['Скорость проходки(м/ч)'],
        #                 index=data_new_interval.index)
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
        # ts10 = pd.Series(data_new_interval['Расход на входе(л/с)'],
        #                   index=data_new_interval.index)
        ts11 = pd.Series(data_new_interval['Момент на СВП(кН*м)'],
                          index=data_new_interval.index)
        ts12 = pd.Series(data_new_interval['Положение крюкоблока(м)'],
                          index=data_new_interval.index)
        ts13 = pd.Series(data_new_interval['Вес на крюке(тс)'],
                          index=data_new_interval.index)

        # fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines',
        #                           name='Скорость проходки(м/ч)'), row=i + 1, col=1)
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
        # fig.add_trace(go.Scatter(x=ts10.index, y=ts10.values, mode='lines',
        #                           name='Расход на входе(л/с)'), row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=ts11.index, y=ts11.values, mode='lines',
                                  name='Момент на СВП(кН*м)'), row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=ts12.index, y=ts12.values, mode='lines',
                                  name='Положение крюкоблока(м)'), row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=ts13.index, y=ts13.values, mode='lines',
                                  name='Вес на крюке(тс)'), row=i + 1, col=1)
    elif i == 1:

        # ts = pd.Series(data_new_interval['Скорость проходки(м/ч)'],
        # index=data_new_interval.index)
        # ts1 = pd.Series(data_new_interval['Момент на ключе(кН*м)'],
        #                 index=data_new_interval.index)
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
        # ts9 = pd.Series(data_new_interval['Момент на маш.ключе(кН*м)'],
        #                 index=data_new_interval.index)
        # ts10 = pd.Series(data_new_interval['Момент на маш.ключе(кН*м).1'],
        #                   index=data_new_interval.index)
        # ts11 = pd.Series(data_new_interval['Скорость СПО(м/с)'],
        #                   index=data_new_interval.index)
        # ts12 = pd.Series(data_new_interval['Положение крюкоблока(м)'],
        # index=data_new_interval.index)
        # ts13 = pd.Series(data_new_interval['Вес на крюке(тс)'],
        # index=data_new_interval.index)

        # fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines',
        #                          name='Скорость проходки(м/ч)'), row=i+1, col=1)
        # fig.add_trace(go.Scatter(x=ts1.index, y=ts1.values, mode='lines',
                                  # name='Момент на ключе(кН*м)'), row=i + 1, col=1)
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
        # fig.add_trace(go.Scatter(x=ts9.index, y=ts9.values, mode='lines',
        #                           name='Момент на маш.ключе(кН*м)'), row=i + 1, col=1)
        # fig.add_trace(go.Scatter(x=ts10.index, y=ts10.values, mode='lines',
        #                           name='Момент на маш.ключе(кН*м).1'), row=i + 1, col=1)
        # fig.add_trace(go.Scatter(x=ts11.index, y=ts11.values, mode='lines',
                                  # name='Скорость СПО(м/с)'), row=i + 1, col=1)
        # fig.add_trace(go.Scatter(x=ts12.index, y=ts12.values, mode='lines',
        # name='Положение крюкоблока(м)'), row=i+1, col=1)
        # fig.add_trace(go.Scatter(x=ts13.index, y=ts13.values, mode='lines',
        # name='Вес на крюке(тс)'), row=i+1, col=1)
    elif i == 2:

        # ts = pd.Series(data_new_interval['Расход на входе(л/с)'],
        #                 index=data_new_interval.index)
        ts1 = pd.Series(data_new_interval['Ходы насоса(ход/мин)'],
                        index=data_new_interval.index)
        ts2 = pd.Series(data_new_interval['Ходы насоса(ход/мин).1'],
                        index=data_new_interval.index)
        ts3 = pd.Series(data_new_interval['Давление в манифольде(МПа)'],
                        index=data_new_interval.index)
        ts4 = pd.Series(data_new_interval['Температура окр.среды(C)'],
                        index=data_new_interval.index)

        # fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines',
        #                           name='Расход на входе(л/с)'), row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=ts1.index, y=ts1.values, mode='lines',
                                  name='Ходы насоса(ход/мин)'), row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=ts2.index, y=ts2.values, mode='lines',
                                  name='Ходы насоса(ход/мин).1'), row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=ts3.index, y=ts3.values, mode='lines',
                                  name='Давление в манифольде(МПа)'), row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=ts4.index, y=ts4.values, mode='lines',
                                  name='Температура окр.среды(C)'), row=i + 1, col=1)
    elif i == 3:
        fig.add_trace(go.Scatter(x=binary_target.index, y=binary_target.values, mode='lines', name='binary_target',
                                  line=dict(color='green')), row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=multi_target.index, y=multi_target.values, mode='lines', name='multi_target',
                                  line=dict(color='red')), row=i + 1, col=1)

for i in range(3):
    # Add shapes for background zones
    for j in range(len(y_zones) - 1):
        fig.add_shape(
            type="rect",
            x0=data_new_interval.index[0],
            y0=y_zones[j],
            x1=data_new_interval.index[-1],
            y1=y_zones[j + 1],
            fillcolor=zone_colors[j],
            layer="below",
            line=dict(color="rgba(0, 0, 0, 0)")
            , row=i+1, col=1
        )
    # Add horizontal lines at 20 and 40
    for y in [20, 40]:
        fig.add_shape(type="line",
                      x0=data_new_interval.index[0], y0=y,
                      x1=data_new_interval.index[-1], y1=y,
                      line=dict(color="rgba(0,0,0,0.5)", width=1, dash="dash"),
                      row=i+1, col=1
                      )
    # Add vertical lines
    for line in vertical_lines_start:
        fig.add_shape(type="line",
                      x0=line, y0=0,
                      x1=line, y1=1000,
                      line=dict(color="rgba(1,41,105,1)", width=2, dash="dash"),
                      row=i+1, col=1
                      )
    # Add vertical lines
    for line in vertical_lines_end:
        fig.add_shape(type="line",
                      x0=line, y0=0,
                      x1=line, y1=1000,
                      line=dict(color="rgba(190,11,49,1)", width=2, dash="dash"),
                      row=i+1, col=1
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
    title=f'Показатели бурения нефтяных скважин',
    showlegend=True,
    # height=900
)
# and legends for each subplot
# fig.update_annotations({'xref': 'paper', 'yref': 'paper', 'x': 0.5, 'y': 1.15, 'showarrow': False, 'text': 'Plot 1'}, row=1, col=1)
# fig.update_annotations({'xref': 'paper', 'yref': 'paper', 'x': 0.5, 'y': 0.78, 'showarrow': False, 'text': 'Plot 2'}, row=2, col=1)
# fig.update_annotations({'xref': 'paper', 'yref': 'paper', 'x': 0.5, 'y': 0.45, 'showarrow': False, 'text': 'Plot 3'}, row=3, col=1)
# Show plot
fig.show()

# df_interpolated.loc['2023-11-07 00:00:00':'2023-11-08 00:00:00', 'WithinInterval'] = 1
#
# df_interpolated.loc['2023-11-07 00:00:00':'2023-11-08 00:00:00', 'WithinInterval'] = df_interpolated.loc['2023-11-07 00:00:00':'2023-11-08 00:00:00', 'Target']