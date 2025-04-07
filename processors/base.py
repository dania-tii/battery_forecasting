import io
import json
import os
import pickle
import re
import zipfile
import logging
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
import requests
import pandas as pd

from time import time
from tqdm import tqdm
from hashlib import md5
from urllib.parse import unquote
from typing import Callable, List, Optional, Tuple, Type, TypeVar, Union, overload

class BaseDataset:
    _DEFAULT_DATASET_URL = ""

    # Google Drive
    _G_FOLDER_PATT = re.compile(r"(?:https?://)?drive\.google\.com/.*?/folders/([^/?#]+)")
    _google_api = None
    _G_DRIVE_FILTERS = []

    T = TypeVar("T", bound="BaseDataset")

    @classmethod
    def _make_logger(cls, name: str) -> logging.Logger:       
        formatter = logging.Formatter("%(name)s::%(levelname)s - %(message)s")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        log = logging.getLogger(name)

        if not log.hasHandlers():
            log.addHandler(handler)

        log.setLevel(logging.DEBUG)
        return log

    def __init_logger__(self):
        if hasattr(self, "log"):
            return self.log
        self.log = self._make_logger(self.__class__.__name__)

    @classmethod
    def __class_asserts__(cls):
        assert getattr(cls, "CAPACITY"), "Battery Capacity is not specified"
        assert getattr(cls, "VOLTAGE_MIN"), "Minimum Voltage is not specified"
        assert getattr(cls, "VOLTAGE_MAX"), "Maximum Voltage is not specified"
        assert getattr(cls, "VOLTAGE_NOM"), "Nominal Voltage is not specified"
        assert getattr(cls, "TIMESTEP_UNIT"), "Timestep Unit is not specified"
        if not cls._DEFAULT_DATASET_URL: print(f"Warning. Empty `_DEFAULT_DATASET_URL`!")

    @classmethod
    def _list_folders(cls, path: str):
        for root, folders, files in os.walk(path):
            yield from folders
            break

    @classmethod
    def _list_files(cls, path: str, filters = []):
        for root, folders, files in os.walk(path):
            if not filters:
                yield from [os.path.join(root, file) for file in files]
            else:
                for file in files:
                    skip = True
                    for filter in filters:
                        if file.endswith(filter):
                            skip = False
                            break
                    if not skip:
                        yield os.path.join(root, file)

    @classmethod
    def _scale_norm(cls, value, min_, max_):
        return (value-min_)/(max_ - min_)

    @classmethod
    def _extract_file_from_content_disposition(cls, string: str):
        match = re.search(r'filename=["\']?([^"\';]+)["\']?', string)
        if match:
            return match.group(1)

        match_star = re.search(r'filename\*=([^;]+)', string)
        if match_star:
            # Decode the filename* using unquote
            encoded_filename = match_star.group(1)
            # Split the encoded filename on the first quote (if exists) to get the value
            # Example: "UTF-8''%E2%82%AC%20rates" -> "€ rates"
            if "''" in encoded_filename:
                _, encoded_filename = encoded_filename.split("''", 1)
            return unquote(encoded_filename)

        return None

    def __init_drive_folders__(self):
        self._drive_folder_ids = {}

    def __init__(self, dataset_path: str, dataset_name: str = ""):
        self.__class_asserts__()
        self.__init_logger__()
        self.__init_columns__()
        self.__init_drive_folders__()
        if not dataset_path:
            raise ValueError(f"Please pass a path to the {self.__class__.__name__} Dataset or a URL to download it from.")

        self.dataset_name = dataset_name or self.__class__.__name__

        if dataset_path.startswith("http"):
            if "drive.google.com" in dataset_path:
                self.dataset_location = self.download_dataset_drive(dataset_path, filters = self._G_DRIVE_FILTERS)
            else:
                self.dataset_location = self.download_dataset(dataset_path)
        elif not os.path.exists(dataset_path):
            raise ValueError(f"Path {dataset_path} does not exist!")
        else:
            self.dataset_location = dataset_path

        self.data = None
        self.lag_amount = 0

        self.log.debug(f"Dataset {self.dataset_name} is in {self.dataset_location}")

    def __init_columns__(self):
        self.columns = {}
        self._warned_covs = True

    def load(self):
        raise NotImplementedError(f"Please implement me!")

    def normalize(self):
        raise NotImplementedError(f"Please implement me!")

    @overload
    def drop_tiny_series(self: T, thresh: int = 1) -> T: ...

    @overload
    def drop_tiny_series(self, thresh: int, df: pd.DataFrame) -> pd.DataFrame: ...

    def drop_tiny_series(self: T, thresh: int = 1, df = None) -> Union[pd.DataFrame, T]:
        return_obj = False
        if df is None:
            df = self.data
            return_obj = True

        counts = df["series"].value_counts() # type: ignore
        valid_series = counts[counts > thresh].index

        output_df: pd.DataFrame = df[df["series"].isin(valid_series)] # type: ignore

        if return_obj:
            self.data = output_df
            return self

        return output_df

    @overload
    def interpolate_missing_timesteps(self: T) -> T: ...

    @overload
    def interpolate_missing_timesteps(self, df: pd.DataFrame) -> pd.DataFrame: ...

    def interpolate_missing_timesteps(self: T, df: Optional[pd.DataFrame] = None) -> Union[pd.DataFrame, T]: # TODO: Spline interp
        return_obj = False
        if df is None:
            return_obj = True
            df = self.data
            if df is None:
                raise ValueError(f"`df` value is missing!")

        min_max: pd.DataFrame = df.groupby("series_time")["timestep"].agg(["min", "max"]).reset_index()  # type: ignore

        all_timesteps = [
            pd.DataFrame({"series_time": st, "timestep": range(row["min"], row["max"] + 1)})
            for st, row in min_max.iterrows()
        ]

        full_timesteps = pd.concat(all_timesteps, ignore_index=True)

        df_filled = full_timesteps.merge(df, on=["series_time", "timestep"], how="left")

        # Interpolate the missing values, preserving duplicates
        df_filled = df_filled.groupby("series_time", group_keys=False).apply(lambda g: g.interpolate(method='linear', axis=0))

        if return_obj:
            self.data = df_filled
            return self

        return df_filled

    @classmethod
    def _recurse_lone_folder(cls, folder_path: str):
        folders = os.listdir(folder_path)
        while len(folders) == 1:
            folder_path = os.path.join(folder_path, folders[0])
            folders = os.listdir(folder_path)
        return folder_path

    @classmethod
    def extract_drive_folder_id(cls, url: str):
        if "drive.google.com" not in url:
            return url

        match = cls._G_FOLDER_PATT.search(url)
        return match.group(1) if match else None

    def setup_g_apis_oauth2(self, config_file: Union[str, dict] = "", creds_file: str = ""):
        if os.path.exists(creds_file):
            with open(creds_file, "rb") as token_file:
                creds = pickle.load(token_file)
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(GoogleRequest())
                    with open(creds_file, "wb") as token_file:
                        pickle.dump(creds, token_file)
                    gdrive_api = build("drive", "v3", credentials=creds)
                    return gdrive_api

        if isinstance(config_file, str):
            with open(config_file, encoding="utf-8") as f:
                config_file = json.load(f)

        scopes = ["https://www.googleapis.com/auth/drive"]

        flow = InstalledAppFlow.from_client_config(config_file, scopes)
        creds = flow.run_local_server(port=18222)

        with open(creds_file, "wb") as token_file:
            pickle.dump(creds, token_file)

        gdrive_api = build("drive", "v3", credentials=creds)
        print("Done?")

    def setup_g_apis(self, config_file: Union[str, dict]):
        scopes = ["https://www.googleapis.com/auth/drive"]
        if isinstance(config_file, str):
            creds = service_account.Credentials.from_service_account_file(config_file, scopes=scopes)
        else:
            creds = service_account.Credentials.from_service_account_info(config_file, scopes=scopes)

        g_api = build("drive", "v3", credentials=creds)
        return g_api

    @property
    def google_api(cls):
        if cls._google_api:
            return cls._google_api
        cls._google_api = cls.setup_g_apis("gdrive_auth/gdrive_config.json")
        return cls._google_api

    def _generate_drive_tree(self, folder_id: str, parent_path = "", filters = [], is_root: bool = True):
        self.log.info(f"Query folder {folder_id} at path {parent_path}")

        if not is_root and folder_id in self._drive_folder_ids:
            self.log.info(f"Skipping {folder_id} since it exists in the list of datasets!")
            return []

        query = f"'{folder_id}' in parents and trashed = false"
        results = self.google_api.files().list(q=query, fields="files(id, name, mimeType)").execute()
        files = results.get("files", [])

        download_list = []

        for file in files:
            file_name = file["name"]
            file_id = file["id"]
            file_mime = file["mimeType"]
            relative_path = os.path.join(parent_path, file_name)

            if not file_mime.endswith(".folder"): # file
                skip = True
                for filter in filters:
                    if file_name.endswith(filter):
                        skip = False
                        break
                if skip:
                    continue
                download_list.append({"id": file_id, "path": parent_path, "name": file_name})
            else:
                subfolder_files = self._generate_drive_tree(file_id, relative_path, filters, False)
                download_list.extend(subfolder_files)

        return download_list

    def _get_drive_folder_name(self, folder_id: str):
        folder_meta = self.google_api.files().get(fileId=folder_id, fields="name").execute()
        folder_name = folder_meta["name"]
        return folder_name

    def download_dataset_drive(self, url: str, output_folder: str= "", filters=[]):     
        folder_id = self.extract_drive_folder_id(url)
        if not folder_id:
            raise ValueError(f"Invalid Google Drive Folder id: {folder_id}")

        dataset_folder = os.path.join(output_folder or "data", self.dataset_name, folder_id)

        for *_, files in os.walk(dataset_folder):
            for file in files:
                if filters:
                    for filter in filters:
                        if file.endswith(filter):
                            return dataset_folder # Exists, don't redownload
                else:
                    if len(file.rsplit(".", 1)) > 1:
                        return dataset_folder # Files exist, no need to re-download

        folder_name = self._get_drive_folder_name(folder_id)

        filtered_download_list = self._generate_drive_tree(folder_id, folder_name, filters=filters)
        for file in filtered_download_list:
            file_name = file["name"]
            file_id = file["id"]
            file_path = file["path"]
            download_path = os.path.join(dataset_folder, file_path)
            try:
                self._download_drive_file(file_id, file_name, download_path)
            except Exception as e:
                self.log.warning(f"Failed to download file {file_name} ({file_id})")

        return dataset_folder

    def _download_drive_file(self, file_id: str, file_name: str, output_dir: str):
        request = self.google_api.files().get_media(fileId=file_id)
        os.makedirs(output_dir, exist_ok=True)

        file_path = os.path.join(output_dir, file_name)
        print(f"Downloading {file_name}...", end="\r")
        with io.FileIO(file_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk(num_retries=3)
                print(f"Downloading {file_name}: {int(status.progress() * 100)}%...", end="\r")
            self.log.info(f"Downloading {file_name}: Complete!")

    def download_dataset(self, url: str, output_folder: str = "", chunk_size: int = 4096):
        dataset_name = self.dataset_name
        output_folder = output_folder or "data"
        dataset_folder = os.path.join(output_folder, dataset_name)
        os.makedirs(dataset_folder, exist_ok=True)
        state = None

        file_name = os.path.basename(url)
        file_ext = os.path.splitext(file_name)[-1]
        if not file_ext.strip():
            file_name = self.dataset_name
        file_location = os.path.join(dataset_folder, file_name)

        if os.path.isfile(file_location):
            state = "downloaded" # Downloaded file exists

        if os.path.isdir(dataset_folder):
            files = os.listdir(dataset_folder)
            if len(files) == 1 and files[0] == file_name:
                state = "downloaded" # Only file that exists is the downloaded file
            elif len(files) == 1 and files[0].rsplit(".", 1)[-1].lower().strip(".").strip() == "zip":
                state = "downloaded"
                file_location = os.path.join(dataset_folder, files[0])
            elif len(files) == 0: # Impossible
                state = None
            else:
                state = "extracted"
                return self._recurse_lone_folder(dataset_folder)

        if state != "downloaded":
            self.log.info(f"Dataset {dataset_name} does not exist, downloading...")
            skip = False
            with requests.get(url, stream=True) as resp:
                dataset_size = int(resp.headers.get("content-length", -1))
                dataset_file_name = self._extract_file_from_content_disposition(resp.headers.get("content-disposition", ""))
                if dataset_file_name:
                    self.log.debug(f"Detected file official name! {dataset_file_name}")
                    file_name = dataset_file_name
                    file_location = os.path.join(dataset_folder, file_name)
                    if os.path.isfile(file_location):
                        self.log.info(f"File is already downloaded... skipping...")
                        skip = True
                if not skip:
                    if dataset_size == -1:
                        self.log.warning(f"Warning: Couldn't read file size!")
                    elif dataset_size == 0:
                        raise Exception(f"Received data size == 0!")

                    with open(file_location, "wb") as f, tqdm(total=dataset_size, unit="B", unit_scale=True, desc=file_name) as pbar:
                        for chunk in resp.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
            state = "downloaded"

        try:
            self.log.debug(f"Attempting extract")
            with zipfile.ZipFile(file_location, "r") as zip_ref:
                zip_ref.extractall(dataset_folder)
            os.remove(file_location)
            state = "extracted"
        except zipfile.BadZipFile:
            self.log.debug("No extraction required")
            state = "extracted"

        return self._recurse_lone_folder(dataset_folder)

    def save(self, name: str = "processed", type_ = None, data: Optional[pd.DataFrame] = None, limit: int = 0, mask = None, *mask_call_args):
        data = data or self.data
        if data is None or data.empty:
            raise ValueError(f"There is no data to save!")

        if mask:
            if isinstance(mask, Callable):
                data = data[mask(*mask_call_args)]
            else:
                data = data[mask]
        if data is None or data.empty:
            raise ValueError(f"Masking generated an empty dataframe!")

        if limit > 0:
            self.log.debug("Limiting to %d (%d)", limit, len(data))
            data = data.iloc[:limit]
            self.log.debug(len(data))
        elif limit < 0:
            self.log.debug("Limiting to last %d (%d)", limit, len(data))
            data = data.iloc[limit:]
            self.log.debug(len(data))

        file_name, *ext = name.rsplit(".", 1)
        if limit > 0:
            file_name += f"_first{limit}"
        elif limit < 0:
            file_name += f"_last{abs(limit)}"
        if len(ext):
            ext = ext[0]
            if ext.lower() in ["csv", "pkl", "parquet", "np"]:
                type_ = ext.lower()
        if not type_:
            type_ = "parquet"

        type_ = type_.strip().strip(".")
        file_name = f"{file_name}.{type_}"

        dir_out = os.path.join("processed_data", self._get_file_shortname(30))
        os.makedirs(dir_out, exist_ok=True)
        file_out = os.path.join(dir_out, file_name)

        if type_ == "csv":
            data.to_csv(file_out)
        elif type_ == "parquet":
            data.to_parquet(file_out)
        elif type_ == "pkl":
            data.to_pickle(file_out)
        elif type_ == "np":
            data.to_numpy().tofile(file_out)
        else:
            raise NotImplementedError(f"Unsupported type {type_}")

        return file_out

    def _get_file_shortname(self, limit: int = 30):
        _name = os.path.split(self.dataset_location)[-1]
        if len(_name) > limit:
            _name_hash = md5(_name.encode("utf-8")).hexdigest()[:7]
            _name = _name[:limit-7] + _name_hash
        return _name

    def _get_discharge_mask(self):
        return self.data["is_discharge"] == True # type: ignore

    def get_discharge_data(self, load_if_missing: bool = True) -> pd.DataFrame:
        if self.data is None and load_if_missing:
            self.log.warning(f"Data was not loaded. Loading first!")
            self.load()
        df_discharge = self.data[self._get_discharge_mask()]  # type: ignore
        return df_discharge

    @property
    def discharge_data(self):
        return self.get_discharge_data(False)

    def get_data_grouped(self, data=None, groupby="series") -> List[pd.DataFrame]:
        if data is None:
            data = self.data
        elif data == "discharge":
            data = self.get_discharge_data(False)

        return [group for _, group in data.groupby(groupby)]  # type: ignore
    
    def create_lag(self, column: str, data: Optional[pd.DataFrame] = None, lags_amt: Union[int, Tuple[int]] = 1, group_by: str = "series_name"):
        if data is None:
            data = self.data
            if data is None:
                raise ValueError(f"Cannot create lags before data is populated!")
        
        self.log.info(f"Creating {lags_amt} lags for column {column}")
        for lag in range(1, lags_amt + 1) if isinstance(lags_amt, int) else lags_amt:
            for series_name, group in data.groupby(group_by):
                lagged_col = f"{column}_lag_t-{lag}"
                data.loc[group.index, lagged_col] = group[column].shift(lag)
                # data.loc[:lag-1, lagged_col] = data[column].iloc[0]
                data.loc[group.index[:lag], lagged_col] = group[column].iloc[0]
            self.add_covariates(lagged_col)
        

        self.remove_covariate(column)
    
    def create_lags_all(self, data: Optional[pd.DataFrame] = None, lags_amt: Union[int, Tuple[int]] = 1,):
        if data is None:
            data = self.data
            if data is None:
                raise ValueError(f"Cannot create lags before data is populated!")
        
        for target in self.targets:
            if target in self.columns["covariates"]: # Do not use .covariates as it will trigger infinite recursion
                if f"{target}_lag_t-1" in data.columns:
                    continue
                self.create_lag(target, data, lags_amt)
            

    def _get_rounding_dict(self,
        ambient_t: int = 3, battery_t: int = 3,
        current: int = 4, voltage: int = 4,
        discharge: int = 3, soc: int = 2,
        current_n: int = 3, voltage_n: int = 3,
        discharge_n: int = 3, nom_v_n: int = 3,
        px4_estimate_n: int = 4,
        ):
        return {
            "ambient_temperature": ambient_t,
            "battery_temperature": battery_t,
            "current": current,
            "voltage": voltage,
            "discharge_capacity": discharge,
            "soc": soc,
            "soc_alt": soc,
            "current_norm": current_n,
            "voltage_norm": voltage_n,
            "discharge_capacity_norm": discharge_n,
            "nominal_voltage_norm": nom_v_n,
            "px4_estimate": px4_estimate_n,
        }

    def process(self, load_args = {}):
        self.load(**load_args)
        self.normalize()
        if self.lag_amount:
            self.create_lags_all(lags_amt=self.lag_amount)

    def add_statics(self, *statics: str):
        # Add columns that are static
        [self.columns.setdefault("statics", []).append(static) for static in statics]

    def add_covariates(self, *covariates: str):
        [self.columns.setdefault("covariates", []).append(cov) for cov in covariates]
        self._warned_covs = False

    def add_targets(self, *targets: str):
        # Add columns that are targets
        [self.columns.setdefault("targets", []).append(t) for t in targets]

    def set_statics(self, statics: List[str]):
        self.columns["statics"] = statics

    def set_covariates(self, covariates: List[str]):
        self.columns["covariates"] = covariates
        self._warned_covs = False

    def set_targets(self, targets: List[str]):
        self.columns["targets"] = targets

    def set_time_index(self, index: str):
        self.columns["time_index"] = index

    def get_statics(self,):
        return self.columns.get("statics", [])

    def get_covariates(self,):
        covs = self.columns.get("covariates", [])
        if not self._warned_covs:
            self._warn_cov_lags()
        return covs
    
    def get_targets(self,):
        return self.columns["targets"]
    
    def remove_static(self, name: str):
        self._remove_series_item(name, "statics")

    def remove_covariate(self, name: str):
        self._remove_series_item(name, "covariates")

    def remove_target(self, name: str):
        self._remove_series_item(name, "targets")

    
    def _remove_series_item(self, name: str, col: str):
        if name in self.columns[col]:
            self.columns[col].remove(name)
        
    
    def _warn_cov_lags(self):
        for target in self.targets:
            if target in self.columns["covariates"]:
                if self.data is None:
                    continue
                if f"{target}_lag_t-1" in self.data.columns:
                    continue
                self.log.warning(f"Target {target} is also a covariate! It is recommended to create lags! see `self.create_lags`")
        self._warned_covs = True

    def get_time_index(self,) -> str:
        return self.columns["time_index"]

    @property
    def statics(self):
        return self.get_statics()

    @property
    def covariates(self):
        return self.get_covariates()

    @property
    def targets(self):
        return self.get_targets()

    @property
    def time_index(self):
        return self.get_time_index()

    @overload
    def get_static_values(
        self, df: Optional[pd.DataFrame] = None, as_json: bool = False
    ) -> pd.DataFrame: ...

    @overload
    def get_static_values(
        self, df: Optional[pd.DataFrame] = None, as_json: bool = True
    ) -> dict: ...

    def get_static_values(self, df: Optional[pd.DataFrame] = None, as_json: bool = False):
        # Return all the columns that are static
        if df is None:
            df = self.data
        if as_json:
            raise NotImplementedError(f"Statics as JSON Not Implemented")
            return {}

        return df[self.columns["static"]] # type: ignore

    def get_covariate_values(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        # Return all the columns that are static
        if df is None:
            df = self.data
        # return columns that are covariate
        raise NotImplementedError()

    def get_target_values(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if df is None:
            df = self.data

        raise NotImplementedError()

    @classmethod
    def get_default_dataset_url(cls) -> str:
        raise NotImplementedError(f"Default Dataset URL is not specified, please specify manually!")

    @classmethod
    def combine_frames(cls, frames: List[pd.DataFrame], ignore_index: bool = True):
        return pd.concat(frames, ignore_index=ignore_index)

    @classmethod
    def deserialize(cls, save: bool = False, save_discharge_only_data: bool = False, save_limit: int = 0, dataset_url: str = "", **load_args): # load_args are args to pass to `load`
        dataset_url = dataset_url or cls.get_default_dataset_url()
        s_time = time()
        dataset_obj = cls(dataset_url)
        dataset_obj.log.info(f"Converting {cls.__name__} to pd.DataFrame")
        dataset_obj.process(load_args)
        dataset_obj.log.debug(f"Done in {time() - s_time:.3f} seconds!")

        if save or save_discharge_only_data:
            if save:
                s_time = time()
                dataset_obj.log.info("Saved to: %s", dataset_obj.save(type_="csv", limit=save_limit))
                dataset_obj.log.debug(f"Done in {time() - s_time:.3f} seconds!")
            if save_discharge_only_data:
                s_time = time()
                dataset_obj.log.info("Saved to: %s", dataset_obj.save(
                    mask=dataset_obj._get_discharge_mask, name="processed_discharge_only", type_="csv", limit=save_limit
                ))
                dataset_obj.log.debug(f"Done in {time() - s_time:.3f} seconds!")

        return dataset_obj

    @classmethod
    def deserialize_and_combine_df(cls: Type[T], *datasets: str) -> Tuple[T, pd.DataFrame]:
        if not datasets:
            raise ValueError(f"Please pass some dataset names por favor amigo!")

        df = pd.DataFrame()
        for dataset in datasets:
            url = cls.get_available_datasets()[dataset] # type: ignore
            obj = cls.deserialize(dataset=dataset, dataset_url=url)
            df = cls.combine_frames([df, obj.data]) # type: ignore

        return obj, df # type: ignore
