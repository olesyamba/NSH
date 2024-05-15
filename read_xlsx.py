from zipfile import ZipFile
import os
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
import pandas as pd
import numpy as np
import datetime
from datetime import datetime
from zipfile import ZipFile
import psycopg2
import requests
import json
import re
from tqdm import tqdm
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datetime import datetime
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool


# Сброс ограничений на количество выводимых рядов
pd.set_option('display.max_rows', None)
# Сброс ограничений на число столбцов
pd.set_option('display.max_columns', None)
# Сброс ограничений на количество символов в записи
pd.set_option('display.max_colwidth', None)


def scan_folder(parent):
    # iterate over all the files in directory 'parent'
    for file_name in os.listdir(parent):
        if file_name.endswith(".xlsx"):
            # if it's a txt file, print its name (or do whatever you want)
            print(file_name)

dirs = [ '7', '11_451', '11_452'] #

for dir in dirs:
    data = pd.DataFrame()
    parent = fr"C:\Users\olesya.krasnukhina\Documents\Проекты\НСХ\data\rig_{dir}"

    scan_folder(parent)  # Insert parent direcotry's path
    n = 0
    # iterate over all the files in directory 'parent'
    for file_name in os.listdir(parent):
        if file_name.endswith(".xlsx"):

            # if it's a txt file, print its name (or do whatever you want)
            print(f'Open {file_name}')
            data_name = pd.read_excel(f'{parent}/{file_name}', header=0,
                                      # usecols='A, AB',
                                      converters={
                                          'datetime': lambda x: xldate_as_datetime(float(x), 0)
                                      }
                                      )
            data_name.set_index('datetime', inplace=True)
            data_name = data_name.resample('min').mean()
            data_name.reset_index(inplace=True)
            data = pd.concat([data, pd.DataFrame(data_name)])
            print(f'{(n + 1) * 100 / len(os.listdir(parent))} %')
            n += 1

    data.to_csv(f'data/old/data_min_mean_rig_{dir}.csv', index=False)