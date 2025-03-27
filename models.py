from numpy import ndarray, linalg as LA
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor
from tabpfn import TabPFNRegressor

from typing import Tuple, Literal
import itertools
from tqdm import tqdm

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
) -> Tuple[XGBRegressor, Tuple[float], Tuple[float], Tuple[float], Tuple[float]]:
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


def hp_xg_boost_single_task(
    input_data: ndarray,
    output_data: ndarray,
    k: int = 5,
    n_estimators_values: list[int] = [10, 50, 100, 500, 1000],
    max_depth_values: list[int] = [1, 2, 3, 5],
    learning_rate_values: list[float] = [1e-1, 1e-2, 1e-3],
    verbose: bool = False,
) -> Tuple[dict[str, float], dict[str, int], pd.DataFrame]:
    best_r2 = -1
    best_metrics = None
    best_hps = None
    full_results = []

    num_combinations = (
        len(n_estimators_values) * len(max_depth_values) * len(learning_rate_values)
    )
    if verbose:
        print(f"Starting HP search with {num_combinations} combinations")
    for num_est, max_depth, lr in tqdm(
        itertools.product(n_estimators_values, max_depth_values, learning_rate_values),
        total=num_combinations,
    ):
        if verbose:
            print(
                f"Training with n_estimators={num_est}, max_depth={max_depth}, learning_rate={lr}"
            )
        _, _, _, average_mse, average_r2 = train_xgboost_single_task(
            input_data, output_data, k, num_est, max_depth, lr
        )
        full_results.append((num_est, max_depth, lr, average_mse, average_r2))
        if average_r2 > best_r2:
            best_r2 = average_r2
            best_metrics = {"MSE": average_mse, "R2": average_r2}
            best_hps = {"num_est": num_est, "max_d": max_depth, "lr": lr}
    full_results = pd.DataFrame(
        full_results,
        columns=[
            "n_estimators",
            "max_depth",
            "learning_rate",
            "average_mse",
            "average_r2",
        ],
    )
    return best_metrics, best_hps, full_results


def hp_xg_boost_multi_task(
    input_data: ndarray,
    output_data: ndarray,
    k: int = 5,
    n_estimators_values: list[int] = [10, 50, 100, 500, 1000, 3000],
    max_depth_values: list[int] = [1, 2, 3, 5, 10],
    learning_rate_values: list[float] = [1e-1, 1e-2, 1e-3],
    multi_task_strategies: list[Multi_Task_Strategy] = [
        "multi_output_tree",
        "one_output_per_tree",
    ],
    verbose: bool = False,
) -> Tuple[dict[str, float], dict[str, int], pd.DataFrame]:
    best_r2 = -1
    best_metrics = None
    best_hps = None
    full_results = []

    num_combinations = (
        len(n_estimators_values)
        * len(max_depth_values)
        * len(learning_rate_values)
        * len(multi_task_strategies)
    )
    if verbose:
        print(f"Starting HP search with {num_combinations} combinations")
    for num_est, max_depth, lr, strategy in tqdm(
        itertools.product(
            n_estimators_values,
            max_depth_values,
            learning_rate_values,
            multi_task_strategies,
        ),
        total=num_combinations,
    ):
        if verbose:
            print(
                f"Training with n_estimators={num_est}, max_depth={max_depth}, learning_rate={lr}"
            )
        _, _, _, average_mse, average_r2 = train_xgboost_multi_task(
            input_data, output_data, k, num_est, max_depth, lr, strategy
        )
        full_results.append(
            (num_est, max_depth, lr, strategy, *average_mse, *average_r2)
        )
        if LA.norm(average_r2) > LA.norm(best_r2):
            best_r2 = average_r2
            best_metrics = {"MSE": average_mse, "R2": average_r2}
            best_hps = {"num_est": num_est, "max_d": max_depth, "lr": lr}
    hp_columns = ["n_estimators", "max_depth", "learning_rate", "multi_task_strategy"]
    mse_columns = [f"mse_{i}" for i in range(output_data.shape[1])]
    r2_columns = [f"r2_{i}" for i in range(output_data.shape[1])]
    full_results = pd.DataFrame(
        full_results,
        columns=hp_columns + mse_columns + r2_columns,
    )
    return best_metrics, best_hps, full_results
