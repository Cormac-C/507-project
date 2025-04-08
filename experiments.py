from sklearn.utils import Bunch
import pandas as pd
import logging
from datetime import datetime

from models import hp_xg_boost_single_task, hp_xg_boost_multi_task

logger = logging.getLogger(__name__)


def xg_boost_trial(
    data: Bunch,
    normalize: bool = True,
    cross_validation_k: int = 5,
    multi_task: bool = False,
    n_estimators_values: list[int] = [10, 50, 100, 500, 1000, 3000],
    max_depth_values: list[int] = [1, 2, 3, 5],
    learning_rate_values: list[float] = [1e-1, 1e-2, 1e-3],
    verbose: bool = False,
    data_name: str = "data",
) -> tuple[dict, dict, pd.DataFrame]:
    """
    Perform a trial with XGBoost on the given dataset
    Args:
        data (Bunch): DataFrame containing the data
        normalize (bool): Whether to normalize the data
        cross_validation_k (int): Number of folds for cross-validation
        multi_task (bool): Whether to use multi-task learning
        n_estimators_values (list[int]): List of n_estimators values to try
        max_depth_values (list[int]): List of max_depth values to try
        learning_rate_values (list[float]): List of learning_rate values to try
        verbose (bool): Whether to print verbose output
    Returns:
        best_metrics (dict): Dictionary containing the best metrics
        best_hps (dict): Dictionary containing the best hyperparameters
        full_results (pd.DataFrame): DataFrame containing the full results
    """
    X = data.data
    y = data.target

    if normalize:
        X = X - X.mean() / X.std()
        y = y - y.mean() / y.std()

    # Turn to numpy arrays
    X = X.to_numpy()
    y_numpy = y.to_numpy()

    if multi_task:
        logger.info("Running multi-task trial")
        best_metrics, best_hps, full_results = hp_xg_boost_multi_task(
            input_data=X,
            output_data=y_numpy,
            k=cross_validation_k,
            n_estimators_values=n_estimators_values,
            max_depth_values=max_depth_values,
            learning_rate_values=learning_rate_values,
            multi_task_strategies=["multi_output_tree"],
            verbose=verbose,
        )
        logger.info(f"Best metrics: {best_metrics}")
        logger.info(f"Best hyperparameters: {best_hps}")
    else:
        logger.info("Running single-task trial")
        logger.info(f"Number of targets: {y.shape[1]}")
        best_metrics = {}
        best_hps = {}
        full_results_dict = {}
        for target in y.columns:
            logger.info(f"Running trial for target: {target}")
            target_data = y[target].values
            (
                single_target_best_metrics,
                single_target_best_hps,
                single_target_full_results,
            ) = hp_xg_boost_single_task(
                input_data=X,
                output_data=target_data,
                k=cross_validation_k,
                n_estimators_values=n_estimators_values,
                max_depth_values=max_depth_values,
                learning_rate_values=learning_rate_values,
                verbose=verbose,
            )
            logger.info(f"{target} Best metrics: {single_target_best_metrics}")
            logger.info(f"{target} Best hyperparameters: {single_target_best_hps}")
            best_metrics[target] = single_target_best_metrics
            best_hps[target] = single_target_best_hps
            full_results_dict[target] = single_target_full_results
        logger.info(f"Complete best metrics: {best_metrics}")
        logger.info(f"Complete best hyperparameters: {best_hps}")
        # Combine all results into a single DataFrame
        full_results = pd.concat(
            full_results_dict.values(),  # All the DataFrames
            keys=full_results_dict.keys(),  # Adds a hierarchical index with the target name
            names=["target"],
        ).reset_index(level=0)

    # Save the results to a CSV file
    full_results.to_csv(
        f"results/{data_name}_xgboost_results_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv",
        index=False,
    )

    return best_metrics, best_hps, full_results
