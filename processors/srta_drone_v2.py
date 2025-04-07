from typing import Dict, Iterable, List, Literal, Union
from processors.base import BaseDataset
import os
import numpy as np
import pandas as pd
from pyulog import ULog
from collections import Counter

class SRTABatteryDataset(BaseDataset):
    _DEFAULT_DATASET_URL = "https://drive.google.com/drive/folders/1cZgGx7RzTDAG-qZk8w6euj0PuoL5orx5"
    _DEFAULT_DATASET_URL_AD = "https://drive.google.com/drive/folders/1cZgGx7RzTDAG-qZk8w6euj0PuoL5orx5"
    _DEFAULT_DATASET_URL_F4F = "https://drive.google.com/drive/u/0/folders/1Jj8d-u3h4XeO2qh5Lqo9arPcjM93OcfA"
    _DEFAULT_DATASET_URL_JamLocator = "https://drive.google.com/drive/folders/1TmgXuZ8S0Ogi1W7LZ4FaDA6w6JYJ5t3Z"
    _G_DRIVE_FILTERS = [".ulg", ".ulog"]
    CELL_COUNT = 6 # not used anywhere
    CAPACITY = 22200 # was 22000# 3.0 Ah # Approximately every 15a current draws 1mah discharge
    VOLTAGE_MIN = 18 # V # did not update yet
    VOLTAGE_MAX = 25.2 # V # was 25
    VOLTAGE_NOM = 22.2 # V # was 22
    TIMESTEP_UNIT = 60/5 # = 12 # seconds per timestep # Resampled to 1 Hz but it still gives 5 seconds per timestep, and some steps are missing! Since data is stored in minutes, current timestep is 5 points per minute, rougly 1 point for every 12 seconds
    # TODO: check what above is used for

    DATASETS = ["AD", "F4F", "JamLocator"]
    DATASETS_HINT = Literal["AD", "F4F", "JamLocator"]

    def __init__(self, dataset_path: str):
        super().__init__(dataset_path, dataset_name="srta_battery_dataset")
        self.lag_amount = 50

    def __init_columns__(self):
        super().__init_columns__()
        self.set_time_index("timestep") # Reference Time Point
        self.add_targets("voltage_norm") # Targets to predict. 1 = Univariate, 2+ = Multivariate
        # self.add_statics("nominal_voltage_norm",) # Covariates that don't change. Used by model to separate data sequences from each other.
        # TODO: look into other statics to add, capacity, (that 22000 value above, also to add voltage norm (normalized) need to think about normalization applied and also the model used first test covariates
        # self.add_covariates("current_norm", "discharge_capacity_norm", "soc", "voltage_norm") # Covariates that influence the target behavior # Skipping Temperature, Battery Cycle, Discharge Capacity
        self.add_covariates("voltage_norm") # Covariates that influence the target behavior # Skipping Temperature, Battery Cycle, Discharge Capacity
        # TODO: later, Consider adding the temperature(s) into covariates

        # Remove any other columns from being considered as covariates
        # self.remove_covariate("series_name")  # If it exists
        # self.remove_covariate("battery_id")  # If it exists

    def __init_drive_folders__(self):
        super().__init_drive_folders__()
        self._drive_folder_ids = {
            self.extract_drive_folder_id(url)
            for url in (
                self._DEFAULT_DATASET_URL_AD,
                self._DEFAULT_DATASET_URL_F4F,
                self._DEFAULT_DATASET_URL_JamLocator,
            )
        }

    # def create_lag()

    @classmethod
    def get_available_datasets(cls) -> Dict[DATASETS_HINT, str]:
        return {
            "AD": cls._DEFAULT_DATASET_URL_AD,
            "F4F": cls._DEFAULT_DATASET_URL_F4F,
            "JamLocator": cls._DEFAULT_DATASET_URL_JamLocator,
        }

    @property
    def available_datasets(cls):
        return cls.get_available_datasets()

    @classmethod
    def get_default_dataset(cls):
        return "all"

    @property
    def default_dataset_name(cls):
        return cls.get_default_dataset()

    def load_multi(self, datasets: Iterable[DATASETS_HINT]):
        combined_df = pd.DataFrame()
        for d in datasets:
            cur_d = self.load(d)
            combined_df = pd.concat([combined_df, cur_d], ignore_index=True)

        self.data = combined_df

        return self.data

    def _ulog_to_df(
        self, ulog_path: str, messages: list = ["trimmed_battery_status"]
    ) -> Dict[str, pd.DataFrame]:
        """
        Converts a PX4 ULog file to pandas DataFrames.

        Args:
            ulog_path (str): Absolute path to the ULog file.
            messages (list, optional): List of message names to include. Defaults to ["battery_status"].

        Returns:
            Dict[str, pd.DataFrame]: Dictionary of DataFrames, where keys are message names.
        """
        ulog = ULog(ulog_path, messages)
        data = ulog.data_list

        dfs = {}

        counts = Counter(
            d.name.replace("/", "_") for d in data if d.name.replace("/", "_") in messages
        )
        redundant_msgs = {name for name, count in counts.items() if count > 1}

        for d in data:
            msg_name = d.name.replace("/", "_")
            msg_key = f"{msg_name}_{d.multi_id}" if msg_name in redundant_msgs else msg_name

            data_keys = [f.field_name for f in d.field_data if f.field_name != "timestamp"]
            data_keys.insert(0, "timestamp")

            df = pd.DataFrame({k: d.data[k] for k in data_keys})
            dfs[msg_key] = df

        return dfs

    def yield_df_from_path(self, ulog_files):
        for ulog_file in ulog_files:
            raw_name = os.path.splitext(ulog_file)[0]
            # print("raw_name: ", raw_name)
            # quit()
            csv_files = []
            if os.path.isdir(raw_name):
                for f in os.listdir(raw_name):
                    if f.startswith("trimmed_battery_status") and f.endswith(".csv"):
                        csv_files.append(os.path.join(raw_name, f))
            if csv_files:
                largest_csv = max(csv_files, key=os.path.getsize)
                df = pd.read_csv(largest_csv)
            else:
                try:
                    dfs = self._ulog_to_df(ulog_file)
                except Exception as e:
                    print(f"Error {ulog_file}: {e}")
                    continue
                os.makedirs(raw_name, exist_ok=True)

                for name, df in dfs.items():
                    df.to_csv(f"{os.path.join(raw_name, name)}.csv", index=False)

                df = dfs[max(dfs, key=lambda name: len(dfs[name]))]

            yield df, raw_name

    def load(self, dataset: Union[Literal["all"], DATASETS_HINT] = "all"):
        if dataset is None:
            dataset = self.default_dataset_name

        if dataset not in ["all"] + self.DATASETS:
            raise TypeError(f"Unsupported dataset {dataset}")

        if dataset == "all":
            return self.load_multi(self.available_datasets)

        sub_root = os.listdir(self.dataset_location)
        if len(sub_root) != 1:
            raise ValueError(f"Expected exactly 1 folder to be in {self.dataset_location}!")

        cur_dataset_name = sub_root[0]
        dataset_folder = os.path.join(self.dataset_location, cur_dataset_name)

        self.log.info(f"Started loading of dataset {self.dataset_name} - {cur_dataset_name}")

        ulog_files = self._list_files(dataset_folder, filters=[".ulg", ".ulog"])

        # Initialize empty lists to hold all the column data
        timestep_list = []
        battery_id_list = []
        # battery_cycle_list = []
        current_list = []
        voltage_list = []
        ambient_temperature_list = []
        battery_temperature_list = []
        discharge_capacity_list = []
        is_discharge_list = [] # is discharge is when capacity is < previous capacity. If it's = or > then it's charging
        series_list = []
        series_names = []
        soc_list = []
        soc_alt_list = []
        px4_estimate_list = []
        scale_list = []
        series = 0

        for df, df_name in self.yield_df_from_path(ulog_files):
            # print("df_name: ", df_name)
            series_name = '_'.join(df_name.replace("\\", "/").rsplit("/", 2)[1:])

            time_ = np.arange(1, len(df)+1)
            voltage = df["voltage_filtered_v"] # TODO: look into diff between voltage filtered and other
            current = df["current_filtered_a"] # TODO: same as volt, look into diff between curr and curr filtered
            capacity = df["discharged_mah"]
            battery_temperature = df["temperature"]
            ambient_temperature = df["temperature"] # TODO: Replace with sensor_combined or gyro_sensor, as of now other messages are being ignored.
            soc = df["remaining"]
            soc_alt = 1-(df["discharged_mah"]/self.CAPACITY)
            px4_estimate = df["time_remaining_s"]
            scale = df["scale"] # TODO: app some relation/similar trend with discharge, wonder if can be used instead, seems like a normalized discharge
            # print(px4_estimate)
            # quit()
            # battery_cycle = np.full(len(time_), -1) # Unknown
            battery_id = df["id"]
            is_discharge = np.full(len(time_), True)

            series_id = np.full(len(time_), series)
            series_name = np.full(len(time_), series_name)
            series += 1

            timestep_list.append(time_)
            battery_id_list.append(battery_id)
            # battery_cycle_list.append(battery_cycle)
            current_list.append(current)
            voltage_list.append(voltage)
            ambient_temperature_list.append(ambient_temperature)
            battery_temperature_list.append(battery_temperature)
            discharge_capacity_list.append(capacity)
            is_discharge_list.append(is_discharge)
            series_list.append(series_id)
            series_names.append(series_name)
            soc_list.append(soc)
            soc_alt_list.append(soc_alt)
            scale_list.append(scale)
            px4_estimate_list.append(px4_estimate)

        timestep_list = np.concatenate(timestep_list)
        battery_id_list = np.concatenate(battery_id_list)
        # battery_cycle_list = np.concatenate(battery_cycle_list)
        current_list = np.concatenate(current_list)
        voltage_list = np.concatenate(voltage_list)
        ambient_temperature_list = np.concatenate(ambient_temperature_list)
        battery_temperature_list = np.concatenate(battery_temperature_list)
        discharge_capacity_list = np.concatenate(discharge_capacity_list)
        is_discharge_list = np.concatenate(is_discharge_list)
        voltage_nom_list = np.full(len(timestep_list), self.VOLTAGE_NOM) # Maybe store voltage min and max per cycle
        soc_list = np.concatenate(soc_list)
        soc_alt_list = np.concatenate(soc_alt_list)
        series_list = np.concatenate(series_list)
        scale_list = np.concatenate(scale_list)
        series_names = np.concatenate(series_names)
        px4_estimate_list = np.concatenate(px4_estimate_list)

        # Concatenate all lists at once
        # TODO: look into current below, whether it should be divided by timestep unit
        df = pd.DataFrame(
            {
                "series": series_list,
                "series_name": series_names,
                # "series_time": time_series_list,
                "timestep": timestep_list,
                "battery_id": battery_id_list,
                # "battery_cycle": battery_cycle_list,
                "current": current_list, #/self.TIMESTEP_UNIT,  # Sometimes this is /15, sometimes it is /18, so for being safe I will /12 which is timestep
                "voltage": voltage_list,
                "soc": soc_list,
                "soc_alt": soc_alt_list,
                "px4_estimate": px4_estimate_list,
                "nominal_voltage": voltage_nom_list,
                # "ambient_temperature": ambient_temperature_list,
                # "battery_temperature": battery_temperature_list,
                "discharge_capacity": discharge_capacity_list,
                "is_discharge": is_discharge_list,
                "scale": scale_list,
            }
        )

        self.data = df

        return self.data

    # def normalize(self):
    #     inverse_time = 1 / ((self.TIMESTEP_UNIT / 60)) # Should give 5
    #     capacity_perc = ((self.CAPACITY / inverse_time ) / 60)
    #     normals = {
    #         # "current": self.data["current"]/self.CAPACITY*self.TIMESTEP_UNIT,
    #         "current": self.data["current"], # / (self.CAPACITY), # Had to disable normalization because it's not working
    #
    #         # "current": self._scale_norm(
    #         #     self.data["current"], self.VOLTAGE_MIN, self.VOLTAGE_MAX
    #         # ),
    #
    #         # "current": ((self.data["current"]/capacity_perc)/60)*inverse_time, # Complicated formula so bear with me!
    #         # max capcity is 22000 Ah, 22000 / 60 -> 366.66 per minute, /5 -> 73.33 per step.
    #         # In other words, if you use 73.33 per time step then you're gonna run out in exactly 1 hour
    #         # In other words, using 73.33 charge per step is 1 / 5 / 60 of total capacity.
    #         "nominal_voltage": self._scale_norm(
    #             self.VOLTAGE_NOM, self.VOLTAGE_MIN, self.VOLTAGE_MAX
    #         ),  # Reference % of the total scale
    #         "voltage": self._scale_norm(
    #             self.data["voltage"], self.VOLTAGE_MIN, self.VOLTAGE_MAX
    #         ),
    #         "discharge_capacity": self.data["discharge_capacity"], # / (self.CAPACITY),
    #     }
    #     for key, val in normals.items():
    #         self.data[f"{key}_norm"] = val
    #
    #     self.data = self.data.round(self._get_rounding_dict(current=2, current_n=4))
    #     return self


    def normalize(self):
        # Assuming that self.data is a pandas DataFrame
        # Calculate mean and standard deviation for Z-score normalization
        current_mean = self.data['current'].mean()
        current_std = self.data['current'].std()
        discharge_capacity_mean = self.data['discharge_capacity'].mean()
        discharge_capacity_std = self.data['discharge_capacity'].std()

        # Z-score normalization
        # TODO: normalize it using minmax like other, but keep in mind above its already being divided by timestep, that may be messing things up
        self.data['current_norm'] = (self.data['current'] - current_mean) / current_std
        self.data['discharge_capacity_norm'] = (self.data['discharge_capacity'] - discharge_capacity_mean) / discharge_capacity_std

        # # Normalize current using robust scaling (less sensitive to outliers)
        # current_q1 = self.data['current'].quantile(0.25)
        # current_q3 = self.data['current'].quantile(0.75)
        # current_iqr = current_q3 - current_q1
        # self.data['current_norm'] = (self.data['current'] - current_q1) / current_iqr
        #
        # # Normalize discharge capacity between 0 and 1
        # max_discharge = self.data['discharge_capacity'].max()
        # self.data['discharge_capacity_norm'] = self.data['discharge_capacity'] / max_discharge

        # print("self.data['soc'] ", self.data['soc'])
        # print("self.data['px4_estimate'] ", self.data['px4_estimate'])
        # quit()

        # Normalizing voltage based on min and max
        self.data['nominal_voltage_norm'] = self._scale_norm(self.VOLTAGE_NOM, self.VOLTAGE_MIN, self.VOLTAGE_MAX)
        self.data['voltage_norm'] = self._scale_norm(self.data['voltage'], self.VOLTAGE_MIN, self.VOLTAGE_MAX)

        # Optionally round the normalized data
        # self.data = self.data.round({'current_norm': 4, 'discharge_capacity_norm': 4, 'voltage_norm': 4, 'nominal_voltage_norm': 4})

        return self

    # Helper function to scale voltage based on a min-max scaling approach
    def _scale_norm(self, value, min_val, max_val):
        return (value - min_val) / (max_val - min_val)

