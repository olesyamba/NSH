import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
# from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

# Сброс ограничений на количество выводимых рядов
pd.set_option('display.max_rows', None)
# Сброс ограничений на число столбцов
pd.set_option('display.max_columns', None)
# Сброс ограничений на количество символов в записи
pd.set_option('display.max_colwidth', None)


# Define the preprocessing steps and models
# preprocessor = StandardScaler()  # Choose the preprocessor
# preprocessor = MinMaxScaler()
# Инициализация нормализатора
scaler = MinMaxScaler()

df = pd.read_csv('data/data.csv')
y = df[['binary_target']].values

class_counts = df[['binary_target']].value_counts()
scale_pos_weight = class_counts[0] / class_counts[1]
version = f'_5_14'
version_to_write = f'_{str(datetime.now().month)}_{str(datetime.now().day)}'
what_is_new = 'remont_all_factors'
filename = f"Отчет_{what_is_new}_{version}.txt"

models = {
    'Decision Tree' : DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    # 'SVM': SVC(class_weight="balanced",  random_state=42, probability=True),
    'RandomForest': RandomForestClassifier(class_weight="balanced_subsample", random_state=42, n_jobs=-1),
    # 'LightGBM': LGBMClassifier(class_weight="balanced", reg_lambda = 0.5, objective='binary', random_state=42, n_jobs = -1),
    'XGboost' : xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, reg_lambda = 0.5, objective='binary:logistic', random_state=42, n_jobs = -1),
    'CatBoost': CatBoostClassifier(random_state=42, silent=True, iterations=500, loss_function='Logloss', eval_metric='Recall', early_stopping_rounds=20),
    # 'HistGB' : HistGradientBoostingClassifier(n_iter_no_change=3, scoring='roc_auc',class_weight='balanced', random_state=42)
}

# Define hyperparameters grid for each model
# Define hyperparameters grid for each model
param_grids = {
    'Decision Tree' : {'max_depth': [None, 10, 20], 'min_samples_leaf' : [3, 5, 10], 'max_features' : [5, 10, 15] },
    # 'SVM': {'C': [0.1, 1, 10], 'gamma': [0.1, 0.01, 0.001], 'kernel': ['linear', 'rbf']}, # 1, 10, 100
    'RandomForest': {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20]},
    # 'LightGBM': {'boosting_type': ['gbdt', 'rf'], 'learning_rate': [0.01, 0.1, 0.3], 'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20]},
    'XGboost' : {"max_depth": [None, 2, 6, 10, 20], "n_estimators": [50, 100, 200], 'learning_rate': [0.005, 0.01, 0.1, 0.3]},
    'CatBoost': {'max_depth': [None, 2, 6, 10, 20], 'learning_rate':[0.005, 0.01, 0.1, 0.3]},
    # 'HistGB' : {"max_depth": [2, 4, 6], "n_estimators": [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.3]}
}

# Define evaluation metrics
metrics = {
    'Accuracy': accuracy_score,
    'Precision': precision_score,
    'Recall': recall_score,
    'F1': f1_score,
    'ROC-AUC': roc_auc_score
}
columns_need = ['Вес на крюке(тс)',
                'Положение крюкоблока(м)',
                'Момент на СВП(кН*м)', 'Обороты СВП(об/мин)',
                'Расход на входе(л/с)',
              'Температура окр.среды(C)', 'Глубина инструмента(м)',
              'Нагрузка на долото(тс)',  'Наработка каната(т*км)']
              # ,'Ходы насоса(ход/мин)', 'Ходы насоса(ход/мин).1','Глубина забоя(м)',
#               'Давление в манифольде(МПа)','Уровень(м3)', 'Уровень(м3).1', 'Уровень(м3).2', 'Уровень(м3).3'

# Initialize DataFrame to store results
result_test_df = pd.DataFrame(index=models.keys(), columns=metrics.keys())
result_valid_df = result_test_df.copy()

#train min max preprocessor
scaler.fit(df[columns_need])


# 'Расход на входе(л/с)',
#TEST 174 and plot results

test_df = pd.DataFrame(index=models.keys(), columns=metrics.keys())
# ['RandomForest', 'XGboost', 'CatBoost']
# Iterate over models
for model_name, model in models.items():
    if model_name not in ['HistGB']:
        with open(filename, 'a+') as file:
            file.write(f"Модель: {model_name}\n")

        df = pd.read_csv('data/07/prep_data_target_11.csv')  # Split the data into train and test sets
        # df = pd.DataFrame(preprocessor.fit_transform(df[columns_need]))
        # X_test = pd.DataFrame(preprocessor.fit_transform(df[columns_need]), columns=columns_need)
        # df.columns = columns_need
        # X_test = df[columns_need]
        # Нормализация данных
        try:
            df.rename(columns={
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

        scaled_features = scaler.transform(df[columns_need])

        # Конвертация в табличный формат
        X_test = pd.DataFrame(data=scaled_features,
                         columns=df[columns_need].columns)

        X_test.columns = [s.replace(" ", "_") for s in X_test.columns.tolist()]

        column_labels_index = {
            'target': 'binary_target'
        }

        df.rename(columns=column_labels_index, inplace = True)

        # X_test.columns = [s.replace(" ", "_") for s in X_test.columns.tolist()]

        y_test = df[['binary_target']].values

        class_counts = df[['binary_target']].value_counts()
        scale_pos_weight = class_counts[0] / class_counts[1]

        start = datetime.now()
        print(f"Model: {model_name}\n{start}")

        # Correcting X-matrix with results of RFECV
        with open(f'RFECV_{model_name}{version}.pkl', 'rb') as f:
            selector = pickle.load(f)

        selected_features = X_test.columns[selector.support_]
        X_test = X_test[selected_features]

        # Importing model
        with open(f'{model_name}{version}.pkl', 'rb') as f:
            model = pickle.load(f)


        # Wrap the classifier in SelectFromModel
        # feature_selector = SelectFromModel(model)
        # Define the pipeline
        # pipeline = Pipeline([
        #     ('preprocessor', preprocessor),
        #     ('feature_selector', feature_selector),
        #     ('model', model)
        # ])

        y_pred = model.predict(X_test)
        log_probs = model.predict_proba(X_test)
        # Convert log probabilities to class predictions
        class_predictions = np.argmax(log_probs, axis=1)

        # Calculate evaluation metrics
        results_test = {}
        for metric_name, metric_func in metrics.items():
            if metric_name == 'ROC-AUC':
                # results[metric_name] = roc_auc_score(y_test, grid_search.best_estimator_.predict_proba(X_test)[:, 1])
                y_pred_proba = log_probs[:, 1]
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                results_test[metric_name] = roc_auc
            elif metric_name == 'Accuracy':
                results_test[metric_name] = metric_func(y_test, y_pred)
            else:
                results_test[metric_name] = metric_func(y_test, class_predictions, average='weighted')


        print('\nTEST RESULTS\n')
        # Update results DataFrame
        test_df.loc[model_name, :] = results_test
        print(test_df.loc[model_name, :])
        test_df = round(test_df, 4).sort_values(by='ROC-AUC', ascending=False)

        with open(filename, 'a+') as file:
            file.write(f"Метрики на тесте по 11 бригаде: \n{test_df.loc[model_name, :]}\n")

        end = datetime.now()
        print(f"\nFULL TIME {model_name} : {end-start}\n\n")


        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(pd.DataFrame(conf_matrix))

        with open(filename, 'a+') as file:
            file.write(f"Confusion matrix: \n{conf_matrix}\n")

        # Setting the attributes
        fig, px = plt.subplots(figsize=(7.5, 7.5))
        px.matshow(conf_matrix, cmap=plt.cm.YlOrRd, alpha=0.5)
        for m in range(conf_matrix.shape[0]):
            for n in range(conf_matrix.shape[1]):
                px.text(x=m, y=n, s=conf_matrix[m, n], va="center", ha="center", size="xx-large")

        # Sets the labels
        plt.xlabel("Actuals", fontsize=16)
        plt.ylabel("Predictions", fontsize=16)
        plt.title(f"Confusion Matrix | {model_name}", fontsize=15)
        plt.show()

        # disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels = model.classes_)
        # disp.plot()
        # # Sets the labels
        # plt.xlabel("Predictions", fontsize=16)
        # plt.ylabel("Actuals", fontsize=16)
        # plt.title(f"Confusion Matrix | {model_name}", fontsize=15)
        # plt.show()

        # Plot ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, log_probs[:, 1])

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) Curve | {model_name}')
        plt.legend()
        plt.show()

        # Assuming features are in X_train (your training data)
        feature_importance = pd.DataFrame({'Feature': X_test.columns, 'Importance': model.feature_importances_})
        sorted_importance = feature_importance.sort_values(by='Importance', ascending=False)


        # Plotting Feature Importance
        plt.figure(figsize=(10, 6))
        plt.barh(sorted_importance['Feature'], sorted_importance['Importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Feature Importance | {model_name}')
        plt.show()

test_df = test_df.astype('float64').apply(lambda x: round(x, 4))

