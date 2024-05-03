from zipfile import ZipFile

from pyforest import *
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import datetime
# Сброс ограничений на количество выводимых рядов
pd.set_option('display.max_rows', None)
# Сброс ограничений на число столбцов
pd.set_option('display.max_columns', None)
# Сброс ограничений на количество символов в записи
pd.set_option('display.max_colwidth', None)


# reading excel with converting excel dates to datetime
# Example usage
import pandas as pn
from xlrd.xldate import xldate_as_datetime

#sql
#Подключаемся к кликхаусу

connection_default = {'host': 'https://clickhouse.lab.karpov.courses',
                      'database':'default',
                      'user':'student',
                      'password':'dpo_python_2020'
                     }
def select(sql):
    return ph.read_clickhouse(query=sql, connection=connection_default)
# создаем запрос, db автоматически заменится на значение из database
sql = '''
    SELECT st_id,
           toStartOfMonth(timest) as month,
           correct, subject
    FROM {db}.peas   
    '''
#Загрузила для просмотра данные
df_all = select(sql)
df_all.head()




# classic import
df = pd.read_excel('data/rig_7/rig_7_Бригада 7_01.01.2024.xlsx', header=0,
                    converters={
                    'datetime': lambda x: xldate_as_datetime(float(x), 0)
                    }
                  )


# looping opening and concating dataframes
data = pd.DataFrame()
n = 0
with ZipFile('data/rig_11_451.zip') as myzip:
    namelist = myzip.namelist()
    for name in namelist:
        if name in namelist:
            print(f'Open {name}')
            myzip.extract(name)
            data_name = pd.read_excel(f'{name}', header=0, usecols='A, AB',
                    converters={
                    'datetime': lambda x: xldate_as_datetime(float(x), 0)
                    }
                  )
            data_name.set_index('datetime', inplace=True)
            data_name.resample('T').mean()
            data_name.reset_index(inplace=True)
            data = pd.concat([data, pd.DataFrame(data_name)])
            n += 1

# resamplimg by timeline
import pandas as pd

# Assuming your DataFrame is named df and has a column named 'datetime' which contains datetime values
# Setting the 'datetime' column as the index if it's not already the index
df.set_index('datetime', inplace=True)

# Resample the DataFrame to minute frequency and calculate mean
df_minute_mean = df.resample('T').mean()

# Resetting index to convert datetime index back to a column
df_minute_mean.reset_index(inplace=True)


# Display basic information about the dataset
print("Dataset Info:")
print(df.info())

# Display descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check nunique values
print("\nUnique Values:")
print(df.nunique())

#Функция первичного обзора данных
def review(df):
    display(df.head(5))
    print(df.info())
    print('Пропуски:', df.isna().sum())
    print('Явные дубликаты:')
    if df.duplicated().sum() > 0:
        print(df.duplicated().sum())
    else:
        print('Не найдено')

review(df)


#queries
df.query('order_status == "delivered"  & order_approved_at.isnull() & ~order_delivered_customer_date.isnull()').shape
df.query('~order_delivered_customer_date.isnull()')['order_status'].value_counts()
#Посчитаю кол-во покупок у каждого юзера

user_purchases_count  = user_purchases.groupby('customer_unique_id').agg({'order_id':'count'}) \
        .rename(columns={'order_id':'count_purchases'}).sort_values('count_purchases',ascending=False).reset_index()

user_purchases_count.head()

#pivot table
pivot_df_count_not_del_orders = df_count_not_del_orders.pivot_table(
               index='order_purchase_month',
              columns='order_status',
              values='orders',
              aggfunc='sum')

pivot_df_count_not_del_orders.head()
################################################# PLOTS ################################################################


# pip install vaex
import vaex
data.info(memory_usage='deep')
# memory usage: 1.1 GB

file_path = r'C:\Users\olesya.krasnukhina\PycharmProjects\NSH\data\data_11_451_full.csv'
dv = vaex.from_csv(file_path, convert=True, chunk_size=5_000_000)
dv = vaex.open(f'{file_path}.hdf5')
# Correlation matrix
correlation_matrix = df.corr()

# Heatmap for correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

# Pairplot for selected columns
sns.pairplot(df[['column1', 'column2', 'column3', 'column4']])
plt.title('Pairplot of Selected Columns')
plt.show()

# Distribution plots for numerical columns
numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
for column in numerical_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# Boxplot for numerical columns
for column in numerical_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df[column])
    plt.title(f'Boxplot of {column}')
    plt.ylabel(column)
    plt.show()


# ADVANCED PLOTS
for i in df.columns:
    if i != 'target':
        fig = px.box(data_frame = df
            ,x = i
            )
        fig.show()


# Violin plot for numerical columns
for column in numerical_columns:
    plt.figure(figsize=(8, 6))
    sns.violinplot(y=df[column])
    plt.title(f'Violin Plot of {column}')
    plt.ylabel(column)
    plt.show()

# Count plot for categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
for column in categorical_columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x=column)
    plt.title(f'Count Plot of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

# Interactive scatter plot using Plotly
fig = px.scatter(df, x='column1', y='column2', color='column3', size='column4', hover_name='column5',
                 title='Interactive Scatter Plot')
fig.show()



################################ one graph with key time series features ###############################################

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