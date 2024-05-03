from datetime import datetime
# import datetime
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight
import statsmodels.api as sm
import numpy as np
import pickle

list_of_path = ['data/data_11_451_full.csv', 'data/data_rig_11_452_full.csv', 'data/data_rig_7_full.csv']
list_of_path_minutes = [f'data/target_added/minutes_{path.split("/")[1]}' for path in list_of_path]
# list_of_path_seconds = [f'data/target_added/seconds_{path.split("/")[1]}' for path in list_of_path]


# Iterate through a list of paths
for path in list_of_path_minutes:

    # Open CSV file
    data = pd.read_csv(path)

    path = path.split("/")[-1]
    print(f"START {path}: {datetime.now()}")

    # Assuming last column is the target variable
    X = data.iloc[:, 1:-1].fillna(0) # можно попробовать альтернативы покачественнее для обработки пропусков
    y = data.iloc[:, -1].fillna(0)
    print(f"Feature Selection with Recursive Feature Elimination (RFE) {path}: {datetime.now()}")
    # Feature Selection with Recursive Feature Elimination (RFE) random_state=42,
    estimator = LogisticRegression(class_weight = "balanced",  n_jobs=-1, verbose=1)
    # selector = RFECV(estimator, step=1, cv=TimeSeriesSplit(n_splits=5), n_jobs=-1, verbose=1)
    # selector = RFECV(estimator, step=1, cv=StratifiedGroupKFold(n_splits=5), n_jobs=-1, verbose=1)
    # при min_features_to_select = 5 и = 7 самые ок результаты, тестила еще с 10
    selector = RFECV( estimator, min_features_to_select = 5, step=1, cv=TimeSeriesSplit(2, test_size=round(len(X) * 0.2)), n_jobs=-1, verbose=1)
    selector = selector.fit(X, y)
    selected_features = X.columns[selector.support_]

    pickle.dump(selector, open(f'minutes_RFECV_{path}.pkl', 'wb'))

    # Training Logistic Regression Model
    model = sm.Logit(y, sm.add_constant(X[selected_features])).fit()

    # Print model summary
    print(model.summary())

    # Cross-validation with Time Series Split random_state=42,
    # tscv = TimeSeriesSplit(n_splits=5) # получаются фолды с одним классом
    # sgkf = StratifiedGroupKFold(n_splits=5) # в теории вроде бы должен пилить, соблюдая баланс классов, но на скорую руку не заработало
    n_splits = 2
    tscv = TimeSeriesSplit(n_splits, test_size=round(len(X) * 0.2))
    model = LogisticRegression(class_weight = 'balanced', n_jobs=-1, verbose=1)

    # Calculate class weights
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)

    scores = cross_val_score(model, X[selected_features], y, cv=tscv, scoring='accuracy'
                             # ,fit_params={'class_weight': class_weights}
                             )

    print(f"Training Logistic Regression Model {path}: {datetime.now()}")
    # Training Logistic Regression Model
    model.fit(X[selected_features], y)
    pickle.dump(model, open(f'minutes_logit_{path}.pkl', 'wb'))

    # Predict on Test Set
    # Assuming you have a separate test set
    # test_X, test_y = load_test_data()
    # test_predictions = model.predict(test_X)

    # Evaluate Accuracy
    # test_accuracy = accuracy_score(test_y, test_predictions)

    # Print Accuracy
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # Visualize Results
    # Example: Plotting feature importance
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(selector.ranking_) + 1), selector.ranking_)
    plt.show()