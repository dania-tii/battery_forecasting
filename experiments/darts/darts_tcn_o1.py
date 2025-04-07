import os
import argparse
import random
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from pytorch_lightning.callbacks import EarlyStopping
from darts.models import TCNModel, TiDEModel
from darts.metrics import rmse, mape
from darts import TimeSeries
from darts.dataprocessing.transformers import StaticCovariatesTransformer

# Custom imports (from your original script)
from processors.srta_drone_v2 import SRTABatteryDataset
from train.utils import get_series_and_covs, split_test_train_timeseries



def set_random_seed(seed: int = 42):
    """Ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def denormalize(value):
    # below is specifically for voltage
    max_val = 25.2
    min_val = 18
    return value * (max_val - min_val) + min_val

def main():
    parser = argparse.ArgumentParser(
        description="Train battery RUL model with specified train/val/test splits."
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs"
    )
    parser.add_argument(
        "--input-length", type=int, default=50, help="History window size"
    )
    parser.add_argument(
        "--output-length", type=int, default=10, help="Forecast horizon"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    set_random_seed(args.seed)

    # 1) Load All Datasets (AD, F4F, JamLocator)
    datasets = ["F4F", "JamLocator", "AD"]
    # datasets = ["F4F","JamLocator"]
    all_series = []
    all_covs = []

    for dataset_name in datasets:
        ds = SRTABatteryDataset(
            dataset_path=SRTABatteryDataset.get_available_datasets()[dataset_name]
        )
        ds.load(dataset_name)

        ds.normalize()

        # time_series_data = TimeSeries.from_dataframe(ds.data, value_cols=['voltage_norm'])
        #
        # Now apply the transformer to the TimeSeries data
        # transformer = StaticCovariatesTransformer()
        # transformed_series = transformer.fit_transform([time_series_data])

        # After ds.normalize(), add this debug code:
        # print("Data columns:", ds.data.columns.tolist())
        # print("First few rows of data:")
        # print(ds.data.head())

        # # DEBUG: Print all static covariates
        # print("Static covariates being used:", ds._statics)
        #
        # # DEBUG: Check data types of these columns
        # if ds._statics:
        #     print("\nStatic covariate data types:")
        #     for col in ds._statics:
        #         print(f"{col}: {ds.data[col].dtype}")

        # quit()

        series, covs = get_series_and_covs(ds.data, ds, group_by="series_name")

        # Convert series and covariates to float32
        series = [s.astype(np.float32) for s in series]
        covs = [c.astype(np.float32) for c in covs]

        # transformer = StaticCovariatesTransformer()
        # series = transformer.fit_transform(series)

        all_series.extend(series)
        all_covs.extend(covs)

    # Sort series by name for deterministic test selection
    all_series = sorted(all_series, key=lambda ts: ts.components[0])
    all_covs = sorted(all_covs, key=lambda ts: ts.components[0])

    # 2) Split Data into Train, Validation, and Test Sets
    # Use 80% for training, 10% for validation, and 10% for testing
    total_series = len(all_series)
    train_size = int(0.8 * total_series)
    val_size = int(0.1 * total_series)

    train_series = all_series[:train_size]
    train_covs = all_covs[:train_size]

    val_series = all_series[train_size:train_size + val_size]
    val_covs = all_covs[train_size:train_size + val_size]

    test_series = all_series[train_size + val_size:]
    test_covs = all_covs[train_size + val_size:]

    # 3) Model Checkpoint Naming
    model_dir = "experimented_models"
    os.makedirs(model_dir, exist_ok=True)

    model_name = f"TCN_in{args.input_length}_out{args.output_length}_ep{args.epochs}_seed{args.seed}.pt"
    model_path = os.path.join(model_dir, model_name)

    if os.path.isfile(model_path):
        print(f"Found existing model: {model_path}. Loading it...")
        model = TiDEModel.load(model_path)
    else:
        # 4) Create TCN Model with EarlyStopping
        # early_stop = EarlyStopping(
        #     monitor="val_loss", patience=5, min_delta=1e-4, mode="min", verbose=True
        # )
        print(args.epochs)
        model = TiDEModel(
            input_chunk_length=args.input_length,
            output_chunk_length=args.output_length,
            n_epochs=args.epochs,
            random_state=args.seed,
            dropout=0.1,
            pl_trainer_kwargs={
                # "callbacks": [early_stop],
                "accelerator": "cpu",
            },
            use_static_covariates = False
        )

        print("Fitting TiDEModel with early stopping...")
        model.fit(
            series=train_series,
            past_covariates=train_covs,
            val_series=val_series,
            val_past_covariates=val_covs,
            verbose=True,
        )

        print(f"Saving model as {model_path} ...")
        model.save(model_path)

    # 5) Evaluate on Test Set
    print("\nEvaluating on the test set...")

    # Define the portions of the series to use for prediction
    portions = [0.1, 0.3, 0.5, 0.7]  # 10%, 50%, 90%

    # Initialize lists to store RMSE and MAPE for each test series and portion
    test_rmses = {p: [] for p in portions}
    test_mapes = {p: [] for p in portions}

    # Iterate over all test series and evaluate
    for i, (test_ts, test_cov) in enumerate(zip(test_series, test_covs)):
        print(f"\nEvaluating test series {i + 1}/{len(test_series)}...")

        for portion in portions:
            print(f"Predicting using the first {int(portion * 100)}% of the series...")

            # Determine the input length based on the portion
            input_length = int(portion * len(test_ts))

            # Ensure input_length is at least input_chunk_length
            if input_length < args.input_length:
                print(f"Skipping portion {int(portion * 100)}% for series {i + 1} because input_length ({input_length}) < input_chunk_length ({args.input_length})")
                continue

            # Predict the rest of the series
            forecast_test = model.predict(
                n=len(test_ts) - input_length,
                series=test_ts[:input_length],
                past_covariates=test_cov
            )

            # Calculate RMSE and MAPE for this portion
            # actual_test = forecast_test.slice_intersect(test_ts)

            # print("Actual Test Shape:", actual_test.values().shape)
            # print("Forecast Test Shape:", forecast_test.values().shape)

            print('actual_test.time_index: ', test_ts.time_index)
            print('forecast_test.time_index: ', forecast_test.time_index)
            # print("Data used for plotting actual values:", test_ts.pd_dataframe().head())
            # print("Data used for plotting forecast values:", forecast_test.pd_dataframe().head())

            # Extracting data and index from TimeSeries objects
            test_df = test_ts.pd_dataframe()
            forecast_df = forecast_test.pd_dataframe()

            # Denormalizing data
            test_denorm = denormalize(test_df['voltage_norm'].values).flatten()
            forecast_denorm = denormalize(forecast_df['voltage_norm'].values).flatten()

            # Create new DataFrames including the original indices
            test_denorm_df = pd.DataFrame({'voltage_norm': test_denorm}, index=test_df.index)
            forecast_denorm_df = pd.DataFrame({'voltage_norm': forecast_denorm}, index=forecast_df.index)

            # Plot predictions for this portion
            plt.figure(figsize=(12, 6))
            # Plot the entire test series
            plt.plot(test_denorm_df.index, test_denorm_df['voltage_norm'], label='Actual', color='red')
            # Overlay the forecast, ensuring it starts at the correct timestep
            forecast_start_index = test_denorm_df.index.max() - len(forecast_denorm_df) + 1
            forecast_indices = np.arange(forecast_start_index, forecast_start_index + len(forecast_denorm_df))
            plt.plot(forecast_indices, forecast_denorm_df['voltage_norm'], label='Forecast', color='blue', alpha=0.7)
            plt.title(f"Test Forecast vs Actual (Test Series {i + 1}, First {int(portion * 100)}%)")
            plt.xlabel('Timestep')
            plt.ylabel('Voltage Norm')
            plt.legend()
            plt.grid(True)
            plt.show()

            # Slicing DataFrames based on the forecast_start_index
            # Assume forecast_start_index is defined and adjust accordingly
            forecast_start_index = test_df.index.get_loc(forecast_df.index[0])  # Finding the start index in the actual data
            test_denorm_sliced = test_denorm_df.iloc[forecast_start_index: forecast_start_index + len(forecast_denorm)]
            forecast_denorm_sliced = forecast_denorm_df

            # Converting back to TimeSeries
            test_ts_denorm = TimeSeries.from_dataframe(test_denorm_sliced)
            forecast_ts_denorm = TimeSeries.from_dataframe(forecast_denorm_sliced)

            # Now calculate RMSE and MAPE
            test_rmse = round(rmse(test_ts_denorm, forecast_ts_denorm), 4)
            test_mape = round(mape(test_ts_denorm, forecast_ts_denorm), 4)

            print(f"Test RMSE: {test_rmse}")
            print(f"Test MAPE: {test_mape}")

            # Store RMSE and MAPE
            test_rmses[portion].append(test_rmse)
            test_mapes[portion].append(test_mape)

    # Compute overall RMSE and MAPE across all test series for each portion
    for portion in portions:
        if test_rmses[portion]:  # Only compute if there are valid predictions
            overall_rmse = np.mean(test_rmses[portion])
            overall_mape = np.mean(test_mapes[portion])

            print(f"\nOverall Test RMSE (First {int(portion * 100)}%): {overall_rmse:.3f}")
            print(f"Overall Test MAPE (First {int(portion * 100)}%): {overall_mape:.3f}")
        else:
            print(f"\nNo valid predictions for portion {int(portion * 100)}% (input_length < input_chunk_length)")

if __name__ == "__main__":
    main()