from matplotlib import pyplot as plt
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from darts import TimeSeries

from train.utils import splice_by_fraction


def predict_all_at_once(test_series: TimeSeries, covariates: TimeSeries, model: TorchForecastingModel, starting_perc: float = 0, debug_mode: bool = False, debug_voltage: float = 0.5):
    model_input_len = model.input_chunk_length

    start_point, offset_index = splice_by_fraction(test_series, starting_perc, model_input_len)
    preds_amt = len(covariates) - offset_index + model_input_len
    if debug_mode:
        # TESTING LAG THEORY
        covariates_df = covariates.pd_dataframe()
        covariates_df.loc[covariates_df.index >= (model_input_len + 1), 'voltage_norm'] = debug_voltage
        modified_covariates = TimeSeries.from_dataframe(covariates_df)

        preds = model.predict(n=preds_amt, series=start_point, past_covariates=modified_covariates)

        # THEORY CONCLUDED THAT DARTS DOESN'T AUTOMATICALLY LAGS THE AR COVARIATES THEREFORE
        # MANUAL INTERVENTION IS REQUIRED OR REMOVE TARGET FROM COVARIATES
    else:
        preds = model.predict(n=preds_amt, series=start_point, past_covariates=covariates)

    return preds.pd_dataframe() # type: ignore


def predict_all_at_once_tcn(
    test_series: TimeSeries,
    covariates: TimeSeries,
    model: TorchForecastingModel,
    starting_perc: float = 0,
    debug_mode: bool = False,
    debug_voltage: float = 0.5,
):
    model_input_len = model.input_chunk_length

    start_point, offset_index = splice_by_fraction(
        test_series, starting_perc, model_input_len
    )
    preds_amt = len(covariates) - offset_index
    preds = model.predict(
        n=preds_amt, series=start_point, past_covariates=covariates
    )

    return preds.pd_dataframe()  # type: ignore
