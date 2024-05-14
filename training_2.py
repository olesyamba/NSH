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
version = f'_{str(datetime.now().month)}_{str(datetime.now().day)}'
key_metric = 'recall'
what_is_new = 'recall'
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
columns_need = ['Вес на крюке(тс)', 'Давление в манифольде(МПа)',
                'Положение крюкоблока(м)',
                'Момент на СВП(кН*м)', 'Обороты СВП(об/мин)',
                'Расход на входе(л/с)',
              'Температура окр.среды(C)', 'Глубина инструмента(м)',
              'Нагрузка на долото(тс)', 'Глубина забоя(м)', 'Наработка каната(т*км)',
              'Ходы насоса(ход/мин)', 'Ходы насоса(ход/мин).1']
#               'Уровень(м3)', 'Уровень(м3).1', 'Уровень(м3).2', 'Уровень(м3).3'

# Initialize DataFrame to store results
result_test_df = pd.DataFrame(index=models.keys(), columns=metrics.keys())
result_valid_df = result_test_df.copy()

#train min max preprocessor
scaler.fit(df[columns_need])

# Iterate over models
for model_name, model in models.items():

    start = datetime.now()
    print(f"Model: {model_name}\n{start}")
    with open(filename, 'a+') as file:
        file.write(f"Model: {model_name}\n\n")

    df = pd.read_csv('data/data.csv')
    # df.pop('datetime')
    # 'Расход на входе(л/с)',
    # Split the data into train and test sets
    # X = pd.DataFrame(preprocessor.fit_transform(df[columns_need]), columns=columns_need)

    # X.columns = [s.replace(" ", "_") for s in X.columns.tolist()]
    # Передача датасета и преобразование
    scaled_features = scaler.transform(df[columns_need])

    # Конвертация в табличный формат
    X = pd.DataFrame(data=scaled_features,
                     columns=df[columns_need].columns)

    X.columns = [s.replace(" ", "_") for s in X.columns.tolist()]

    y = df[['binary_target']].values

    class_counts = df[['binary_target']].value_counts()
    scale_pos_weight = class_counts[0] / class_counts[1]

    y = np.ravel(y)
    # y_train = df[['multi_target']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Wrap the classifier in SelectFromModel
    # feature_selector = SelectFromModel(model)
    # # Define the pipeline
    # pipeline = Pipeline([
    #     ('preprocessor', preprocessor),
    #     ('feature_selector', feature_selector),
    #     ('model', model)
    # ])

    print(f"Feature Selection with Recursive Feature Elimination (RFE) {datetime.now()}")

    # Perform 5-fold cross-validation to evaluate the preprocessor
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    # min_features_to_select = 5,
    # Feature Selection with Recursive Feature Elimination (RFE) random_state=42,
    selector = RFECV(model, step=1, cv=cv, n_jobs=-1, verbose=1)
    selector = selector.fit(X, y)
    selected_features = X.columns[selector.support_]
    with open(filename, 'a+') as file:
        file.write(f"Отобранные факторы: {selected_features}\n")

    pickle.dump(selector, open(f'RFECV_{model_name}{version}.pkl', 'wb'))

    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    # model = RandomForestClassifier(class_weight="balanced_subsample", warm_start=True, verbose=True, random_state=42, n_jobs=-1)

    # # Define the pipeline
    # pipeline = Pipeline([
    #     ('preprocessor', preprocessor),
    #     ('model', model)
    # ])

    cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'], n_jobs=-1, verbose=1)
    print(f"CV Fit time: {cv_results['fit_time']}")
    print(f"CV Score time: {cv_results['score_time']}")
    print('\nCROSS VALIDATION RESULTS\n')

    # result_valid_df = pd.DataFrame(index=models.keys(), columns = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'])

    result_valid_df.loc[model_name, 'Accuracy'] = round(cv_results['test_accuracy'].mean(), 4)
    result_valid_df.loc[model_name, 'Precision'] = round(cv_results['test_precision'].mean(), 4)
    result_valid_df.loc[model_name, 'Recall'] = round(cv_results['test_recall'].mean(), 4)
    result_valid_df.loc[model_name, 'F1'] = round(cv_results['test_f1'].mean(), 4)
    result_valid_df.loc[model_name, 'ROC-AUC'] = round(cv_results['test_roc_auc'].mean(), 4)
    print(result_valid_df.loc[model_name, :])
    with open(filename, 'a+') as file:
        file.write(f"Метрики на валидации: \n{result_valid_df.loc[model_name, :]}\n")

    # Perform hyperparameter tuning using grid search
    param_grid = param_grids[model_name]
    grid_search = GridSearchCV(model, param_grid, cv=3, refit='recall', scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'], n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters
    print("Best hyperparameters:")
    print(grid_search.best_params_)
    with open(filename, 'a+') as file:
        file.write(f"Значения гиперпараметров: \n{grid_search.best_params_}\n")

    model = grid_search.best_estimator_

    model.fit(X_train, y_train)

    # Predict on the test set
    # y_pred = grid_search.best_estimator_.predict(X_test)

    y_pred = model.predict(X_test)
    # Calculate evaluation metrics
    results = {}
    for metric_name, metric_func in metrics.items():
        if metric_name == 'ROC-AUC':
            # results[metric_name] = roc_auc_score(y_test, grid_search.best_estimator_.predict_proba(X_test)[:, 1])
            results[metric_name] = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        else:
            results[metric_name] = metric_func(y_test, y_pred)

    print('\nTEST RESULTS\n')
    # Update results DataFrame
    result_test_df.loc[model_name, :] = results
    print(result_test_df.loc[model_name, :])
    # result_test_df = result_test_df.apply(round).sort_values(by='ROC-AUC', ascending=False)
    result_test_df = result_test_df.astype('float64').apply(lambda x: round(x, 4)).sort_values(by='ROC-AUC', ascending=False)
    with open(filename, 'a+') as file:
        file.write(f"Метрики на тесте при обучении: \n{result_test_df.loc[model_name, :]}\n")

    # cv_res = pd.DataFrame.from_dict(grid_search.cv_results_)
    # best_id_model = cv_res.sort_values(by='rank_test_roc_auc').head(1).index[0]
    #
    # result_valid_df = pd.DataFrame(index=models.keys(), columns = ['mean_test_accuracy', 'mean_test_precision', 'mean_test_recall', 'mean_test_f1', 'mean_test_roc_auc'])
    #
    # result_valid_df.loc[model_name, 'Accuracy'] = round(cv_res.loc[best_id_model, 'mean_test_accuracy'], 4)
    # result_valid_df.loc[model_name, 'Precision'] = round(cv_res.loc[best_id_model, 'mean_test_precision'], 4)
    # result_valid_df.loc[model_name, 'Recall'] = round(cv_res.loc[best_id_model, 'mean_test_recall'], 4)
    # result_valid_df.loc[model_name, 'F1'] = round(cv_res.loc[best_id_model, 'mean_test_f1'], 4)
    # result_valid_df.loc[model_name, 'ROC-AUC'] = round(cv_res.loc[best_id_model, 'mean_test_roc_auc'], 4)
    end = datetime.now()
    print(f"\nFULL TRAINING TIME {model_name} : {end-start}\n\n")

    pickle.dump(model, open(f'{model_name}{version}.pkl', 'wb'))

result_test_df = result_test_df.astype('float64').apply(lambda x: round(x, 4))
result_valid_df = result_valid_df.astype('float64').apply(lambda x: round(x, 4))

# 'Расход на входе(л/с)',
#TEST 174 and plot results

test_df = pd.DataFrame(index=models.keys(), columns=metrics.keys())
# ['RandomForest', 'XGboost', 'CatBoost']
# Iterate over models
for model_name, model in models.items():
    if model_name not in ['HistGB']:

        df = pd.read_csv('data/prep_data_test_174_new.csv')  # Split the data into train and test sets
        # df = pd.DataFrame(preprocessor.fit_transform(df[columns_need]))
        # X_test = pd.DataFrame(preprocessor.fit_transform(df[columns_need]), columns=columns_need)
        # df.columns = columns_need
        # X_test = df[columns_need]
        # Нормализация данных
        scaled_features = scaler.transform(df[columns_need])

        # Конвертация в табличный формат
        X_test = pd.DataFrame(data=scaled_features,
                         columns=df[columns_need].columns)

        X_test.columns = [s.replace(" ", "_") for s in X_test.columns.tolist()]

        column_labels_index = {
            'target': 'binary_target'
        }

        df.rename(columns=column_labels_index, inplace=True)

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
            file.write(f"Метрики на тесте по 174 скважине: \n{test_df.loc[model_name, :]}\n")

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