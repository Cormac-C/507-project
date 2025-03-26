from numpy import ndarray, linalg as LA

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor
from tabpfn import TabPFNRegressor

from typing import Tuple, Literal

Multi_Task_Strategy = Literal["multi_output_tree", "one_output_per_tree"]

Normalization_Strategy = Literal["none", "standard", "min_max"]


def train_xgboost_single_task(
    input_data: ndarray,
    output_data: ndarray,
    k: int = 5,
    n_estimators: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.1,
) -> Tuple[XGBRegressor, float, float, float, float]:
    models = []
    mses = []
    r2s = []
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    for train_index, test_index in kfold.split(input_data):
        X_train, X_test = input_data[train_index], input_data[test_index]
        y_train, y_test = output_data[train_index], output_data[test_index]
        model = XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        models.append(model)
        mses.append(mse)
        r2s.append(r2)
    best_index = mses.index(min(mses))
    best_model = models[best_index]
    best_mse = mses[best_index]
    best_r2 = r2s[best_index]
    average_mse = sum(mses) / len(mses)
    average_r2 = sum(r2s) / len(r2s)
    return best_model, best_mse, best_r2, average_mse, average_r2


def train_tabpfn_single_task(
    input_data: ndarray, output_data: ndarray, k: int = 5
) -> Tuple[TabPFNRegressor, float, float, float, float]:
    models = []
    mses = []
    r2s = []
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    for train_index, test_index in kfold.split(input_data):
        X_train, X_test = input_data[train_index], input_data[test_index]
        y_train, y_test = output_data[train_index], output_data[test_index]
        model = TabPFNRegressor()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        models.append(model)
        mses.append(mse)
        r2s.append(r2)
    best_index = mses.index(min(mses))
    best_model = models[best_index]
    best_mse = mses[best_index]
    best_r2 = r2s[best_index]
    average_mse = sum(mses) / len(mses)
    average_r2 = sum(r2s) / len(r2s)
    return best_model, best_mse, best_r2, average_mse, average_r2


def train_xgboost_multi_task(
    input_data: ndarray,
    output_data: ndarray,
    k: int = 5,
    n_estimators: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.1,
    multi_task_strategy: Multi_Task_Strategy = "multi_output_tree",
) -> Tuple[XGBRegressor, float, float, float, float]:
    models = []
    mses = []
    r2s = []
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    for train_index, test_index in kfold.split(input_data):
        X_train, X_test = input_data[train_index], input_data[test_index]
        y_train, y_test = output_data[train_index], output_data[test_index]
        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            multi_strategy=multi_task_strategy,
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions, multioutput="raw_values")
        r2 = r2_score(y_test, predictions, multioutput="raw_values")
        models.append(model)
        mses.append(mse)
        r2s.append(r2)
    # Compare norms of mses
    best_index = min(range(len(mses)), key=lambda i: LA.norm(mses[i]))
    best_model = models[best_index]
    best_mse = mses[best_index]
    best_r2 = r2s[best_index]
    average_mse = sum(mses) / len(mses)
    average_r2 = sum(r2s) / len(r2s)
    return best_model, best_mse, best_r2, average_mse, average_r2
