from darts import TimeSeries
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from typing import Iterable, Union, List
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel

import torch


from processors.base import BaseDataset

def __get_timeseries_group(dataset: pd.DataFrame, object: BaseDataset, values: str, group_by: Union[str, List[str]]):
    values = values.lower()
    if values == "targets":
        target = object.targets
    elif values in ["covs", "covariates"]:
        target = object.covariates
    else:
        raise ValueError(f"Unknown values {values}")

    return get_timeseries_by_group(dataset,group_by,time_col=object.time_index,static_cols=object.statics,value_cols=target,fill_missing_dates=False,freq=1,)

def get_timeseries_by_group(dataset, group_by, time_col, value_cols, static_cols=[], fill_missing_dates=False, freq=1):
    return TimeSeries.from_group_dataframe(dataset,group_by,time_col=time_col,static_cols=static_cols,value_cols=value_cols,fill_missing_dates=fill_missing_dates,freq=freq,)

def get_series_and_covs(
    dataset: pd.DataFrame,
    object: BaseDataset,
    group_by: Union[str, List[str]] = "series",
):
    # Need a way to maintain the names since it is being shuffled somehow
    series = __get_timeseries_group(dataset, object, "targets", group_by)
    covariates = __get_timeseries_group(dataset, object, "covariates", group_by)

    return series, covariates

def split_test_train_timeseries(series, cov, split: float = 0.9):
    train_size = int(len(series) * split)
    train_series = series[:train_size]
    train_cov = cov[:train_size]
    test_series = series[train_size:]
    test_cov = cov[train_size:]

    return train_series, train_cov, test_series, test_cov

def rmse_loss(y_pred, y_true):
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))


def plot_all(
    series,
    title: str = "",
    show=False,
    save_file_path=None,
    blocking=True,
    x_label: str = "",
    y_label: str = "",
):
    for i, d in enumerate(series):
        to_label = f"{i+1}"
        if not i:
            d.plot(True, label=to_label)
        else:
            d.plot(False, label=to_label)

    if title:
        plt.title(title)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    plt.legend()

    if save_file_path:
        plt.savefig(save_file_path, dpi=300)

    if show:
        if blocking:
            plt.show()
        else:
            plt.gcf().show()

def predict_until_zero_cov(
    test_series: TimeSeries, covariates: TimeSeries, model, max_attempts=100
):
    model_input_len = model.input_chunk_length
    model_output_len = model.output_chunk_length

    initial_series = test_series[:model_input_len]
    initial_cov = covariates[:model_input_len]

    current_series = initial_series[:]
    current_cov = initial_cov[:]

    last_soc = 1

    cur_attempts = -1

    while last_soc >= 0.05:
        cur_attempts += 1
        if cur_attempts > max_attempts:
            break
        forecast = model.predict(
            n=model_output_len,
            series=current_series,
            past_covariates=current_cov,
        )
        forecast_values: ndarray = forecast.values()[:, 0]  # type: ignore

        # Calc covs
        new_current_norm = np.tile(
            initial_cov["current_norm"].values(),
            (model_output_len // len(initial_cov["current_norm"].values()) + 1, 1),
        )[
            :model_output_len
        ]  # Create the first 10 only, later append to original 5
        new_voltage_norm = forecast_values.reshape(-1, 1)

        last_discharge_capacity = current_cov["discharge_capacity_norm"].values()[
            -1
        ]
        new_discharge_capacity_norm = np.array(
            [
                last_discharge_capacity - sum(new_current_norm[: i + 1])
                for i in range(model_output_len)
            ]
        ).reshape(-1, 1)

        new_cov_values = np.hstack(
            [new_current_norm, new_voltage_norm, new_discharge_capacity_norm]
        )
        new_cov = TimeSeries.from_times_and_values(
            forecast.time_index,
            new_cov_values,
            columns=["current_norm", "voltage_norm", "discharge_capacity_norm"],
        )

        current_series = current_series.append(forecast)  # type: ignore # len + 10, so need to generate 10 covs
        current_cov = current_cov.append(new_cov)
        last_soc = min(forecast_values)

        # TODO: Check if I need to just take the last values or pass all of it by observing the outputs, or simply just splicing on predict input. I need to see all the values.
        # Answer: Only values

    return current_series, current_cov

def predict_feedback(
    test_series: TimeSeries, covariates: TimeSeries, model: TorchForecastingModel
):
    prediction_length = model.output_chunk_length
    input_length = model.input_chunk_length

    covariates_df = covariates.pd_dataframe()
    lagged_columns = [col for col in covariates_df.columns if "_lag_t-" in col]

    predictions = []
    start_point = get_split_fraction(test_series, 0, input_length)
    preds_amt = len(covariates) - start_point
    current_series = test_series[start_point : start_point + input_length]

    while start_point + prediction_length < preds_amt:
        cur_predicts: TimeSeries = model.predict(n=prediction_length, series=current_series, past_covariates=covariates)  # type: ignore
        predictions.extend(cur_predicts.values()[:, 0])

        # Update the covariates_df variable with the values from cur_predicts
        pred_df = cur_predicts.pd_dataframe()

        for lagged_col in lagged_columns:
            original_col, lag = lagged_col.split("_lag_t-", 1)
            lag = int(lag)

            if original_col in pred_df.columns:
                insert_idx = (
                    start_point + input_length + lag + 1
                )  # Unsure if this must be + 1 or not
                end_idx = insert_idx + prediction_length
                if insert_idx < len(covariates_df):
                    insert_length = len(
                        covariates_df.loc[insert_idx : end_idx - 1, lagged_col]
                    )
                    covariates_df.loc[insert_idx : end_idx - 1, lagged_col] = (
                        pred_df[original_col].values[:insert_length]
                    )

        covariates = TimeSeries.from_dataframe(covariates_df)

        start_point += prediction_length
        
        # This only works if output length >= input length. If input length > output_length then you need to pad by previous values
        current_series = cur_predicts  # Use this if you want to continue from prediction_length (t+10)
        # current_series = cur_predicts[:input_length] # Use this if you want to continue from input length (t+5)

    return pd.DataFrame(predictions)


def get_model_name(model, class_, datasets: Iterable = []):
    name = f"{class_.__name__}-{model.__class__.__name__}-num_layers={model.num_layers}-layer_widths={model.layer_widths[0]}-history_length={model.input_chunk_length}-horizon_length={model.output_chunk_length}"
    if datasets:
        name += '.data.' + '.'.join(datasets)
    return name


def splice_by_fraction(data, fraction, length):
    start = get_split_fraction(data, fraction, length)
    return data[start: start + length], start + length


def get_split_fraction(data, fraction, length):
    if not (0 <= fraction < 1):
        raise ValueError(f"Fraction must be ∈ [0, 1)")

    start = int(len(data) * fraction)

    if start + length > len(data):
        raise ValueError(
            f"Invalid length: starting at index {start} with length {length} exceeds data size {len(data)}."
        )

    return start
