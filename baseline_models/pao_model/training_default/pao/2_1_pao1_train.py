EXP_ID = "272"
GPU_IX = 0
LIGHT_VERSION = False

from pathlib import Path
import logging

EXP_DIR = Path(f"../data/exp{EXP_ID}")
EXP_DIR.mkdir(exist_ok=True, parents=True)
INPUT_DIR = Path("../input")


# Logger setup
logger = logging.getLogger('exp_logger')
logger.setLevel(logging.DEBUG)

# File handler
fh = logging.FileHandler(EXP_DIR / 'experiment.log')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

# Stream handler
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(fh)
logger.addHandler(sh)


import os
import re
import gc
import time
import pickle
import random
from pathlib import Path
from pprint import pprint
from collections import Counter, defaultdict
from typing import Union
from copy import deepcopy

from collections import OrderedDict

import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import kurtosis

import torch
from torch import nn
from torch.cuda import amp
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    ExponentialLR
)
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

import transformers
import webdataset as wds

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)


class CFG:
    mode = "cv_and_all"
    seed = 2024
    n_folds = 4

    num_workers = 4
    batch_size = 256
    n_epoch = 10
    lr = 1e-3
    SchedulerClass = CosineAnnealingLR
    scheduler_params = dict(
        T_max=n_epoch,
        eta_min=1e-5
    )
    weight_decay = 1.0
    verbose = True
    verbose_step = 1
    num_warmup_steps_rate = 0.0
    use_fp16 = False


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_data():
    train_df = pl.read_parquet(INPUT_DIR / "new_train.parquet")
    test_df = pl.read_parquet(INPUT_DIR / "new_test.parquet")
    old_weight = pl.read_csv(INPUT_DIR / "sample_submission_old.csv", n_rows=1).to_dict(as_series=False)
    old_weight = {k: v[0] for k, v in old_weight.items()}
    new_weight = pl.read_csv(INPUT_DIR / "sample_submission.csv", n_rows=1).to_dict(as_series=False)
    new_weight = {k: v[0] for k, v in new_weight.items()}
    return train_df, test_df, old_weight, new_weight


FEATURE_SEQ_GROUPS = [
    "state_q0001",
    "state_q0002",
    "state_q0003",
    "state_t",
    "state_u",
    "state_v",
    "pbuf_ozone",
    "pbuf_N2O",
    "pbuf_CH4",
    "rel_vapor_pressure"
]
FEATURE_SCALER_COLS = [
    'state_ps',
    'pbuf_SOLIN',
    'pbuf_LHFLX',
    'pbuf_SHFLX',
    'pbuf_TAUX',
    'pbuf_TAUY',
    'pbuf_COSZRS',
    'cam_in_ALDIF',
    'cam_in_ALDIR',
    'cam_in_ASDIF',
    'cam_in_ASDIR',
    'cam_in_LWUP',
    'cam_in_ICEFRAC',
    'cam_in_LANDFRAC',
    'cam_in_OCNFRAC',
    'cam_in_SNOWHLAND',
]
TARGET_SEQ_GROUPS = [
    'ptend_t',
    'ptend_u',
    'ptend_v',
    'ptend_q0001',
    'ptend_q0002',
    'ptend_q0003'
]
TARGET_SCALER_COLS = [
    'cam_out_NETSW',
    'cam_out_FLWDS',
    'cam_out_PRECSC',
    'cam_out_PRECC',
    'cam_out_SOLS',
    'cam_out_SOLL',
    'cam_out_SOLSD',
    'cam_out_SOLLD'
]

class ModelEmaV2(nn.Module):
    """ Model Exponential Moving Average V2

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

def add_seq_features(train_df, test_df, val_sprit=None, valid_num=None, all_train=False):
    if not all_train:
        train_seq_features = np.zeros((len(train_df) - valid_num, len(FEATURE_SEQ_GROUPS), 60))
        valid_seq_features = np.zeros((valid_num, len(FEATURE_SEQ_GROUPS), 60))
    else:
        train_seq_features = np.zeros((len(train_df), len(FEATURE_SEQ_GROUPS), 60))

    test_seq_features = np.zeros((test_df.shape[0], len(FEATURE_SEQ_GROUPS), 60))

    for group_ix, group in enumerate(FEATURE_SEQ_GROUPS):
        cols = [f"{group}_{i}" for i in range(60)]
        if not all_train:
            train_seq_features[:, group_ix] += train_df.select(cols).to_numpy()[~val_sprit]
            valid_seq_features[:, group_ix] += train_df.select(cols).to_numpy()[val_sprit]
        else:
            train_seq_features[:, group_ix] += train_df.select(cols).to_numpy()
        test_seq_features[:, group_ix] += test_df.select(cols).to_numpy()
        train_df = train_df.drop(cols)
        test_df = test_df.drop(cols)
    gc.collect()
    logger.info("Finished adding sequence features")
    if not all_train:
        return train_seq_features, valid_seq_features, test_seq_features, train_df, test_df
    else:
        return train_seq_features, test_seq_features, train_df, test_df


def add_scaler_features(train_df, test_df, val_sprit=None, valid_num=None, all_train=False):
    if not all_train:
        train_scaler_features = train_df.select(FEATURE_SCALER_COLS).to_numpy()[~val_sprit]
        valid_scaler_features = train_df.select(FEATURE_SCALER_COLS).to_numpy()[val_sprit]
    else:
        train_scaler_features = train_df.select(FEATURE_SCALER_COLS).to_numpy()
    test_scaler_features = test_df.select(FEATURE_SCALER_COLS).to_numpy()
    train_df = train_df.drop(FEATURE_SCALER_COLS)
    test_df = test_df.drop(FEATURE_SCALER_COLS)
    gc.collect()
    logger.info("Finished adding scaler features")
    if not all_train:
        return train_scaler_features, valid_scaler_features, test_scaler_features, train_df, test_df
    else:
        return train_scaler_features, test_scaler_features, train_df, test_df


def add_seq_targets(train_df, val_sprit, valid_num):
    train_seq_targets = np.zeros((len(train_df) - valid_num, len(TARGET_SEQ_GROUPS), 60))
    valid_seq_targets = np.zeros((valid_num, len(TARGET_SEQ_GROUPS), 60))
    for group_ix, group in enumerate(TARGET_SEQ_GROUPS):
        cols = [f"{group}_{i}" for i in range(60)]
        train_seq_targets[:, group_ix] += train_df.select(cols).to_numpy()[~val_sprit]
        valid_seq_targets[:, group_ix] += train_df.select(cols).to_numpy()[val_sprit]
        train_df = train_df.drop(cols)
        logger.info(f"Added sequence targets for group: {group}")
    gc.collect()
    logger.info("Finished adding sequence targets")
    return train_seq_targets, valid_seq_targets, train_df


def add_scaler_targets(train_df, val_sprit, valid_num):
    train_scaler_targets = train_df.select(TARGET_SCALER_COLS).to_numpy()[~val_sprit]
    valid_scaler_targets = train_df.select(TARGET_SCALER_COLS).to_numpy()[val_sprit]
    train_df = train_df.drop(TARGET_SCALER_COLS)
    gc.collect()
    logger.info("Finished adding scaler targets")
    return train_scaler_targets, valid_scaler_targets, train_df


def preprocess_data(train_seq_features, valid_seq_features, test_seq_features, train_scaler_features, valid_scaler_features, test_scaler_features,
                    train_seq_targets, valid_seq_targets, train_scaler_targets, valid_scaler_targets, old_weight):
   
    # normalize scaler features
    before_shape = train_scaler_features.shape
    scaler_mean = train_scaler_features.mean(axis=0)
    scaler_std = train_scaler_features.std(axis=0)
    scaler_std = np.maximum(scaler_std, 1e-8)
    train_scaler_features = (train_scaler_features - scaler_mean) / scaler_std
    valid_scaler_features = (valid_scaler_features - scaler_mean) / scaler_std
    test_scaler_features = (test_scaler_features - scaler_mean) / scaler_std
    assert train_scaler_features.shape == before_shape
    logger.info("Finished normalizing scaler features")

    # normalize seq features
    feature_seq_means = {}
    feature_seq_stds = {}
    feature_seq_each_means = {}
    feature_seq_each_stds = {}
    for group_ix, group in enumerate(FEATURE_SEQ_GROUPS):
        before_shape = train_seq_features[:, group_ix, :].shape
        seq_each_mean = train_seq_features[:, group_ix, :].mean(axis=0)
        seq_each_std = train_seq_features[:, group_ix, :].std(axis=0)
        seq_each_std = np.maximum(seq_each_std, 1e-8)
        feature_seq_each_means[group] = seq_each_mean
        feature_seq_each_stds[group] = seq_each_std
        seq_mean = seq_each_mean.mean()
        seq_std = seq_each_std.mean()
        feature_seq_means[group] = seq_mean
        feature_seq_stds[group] = seq_std

    for group_ix, group in enumerate(FEATURE_SEQ_GROUPS):
        if not "state_q000" in group:
            continue
        seq_each_mean = np.log1p(train_seq_features[:, group_ix, :]).mean(axis=0)
        seq_each_std = np.log1p(train_seq_features[:, group_ix, :]).std(axis=0)
        seq_each_std = np.maximum(seq_each_std, 1e-8)
        feature_seq_each_means[group+"_log1p"] = seq_each_mean
        feature_seq_each_stds[group+"_log1p"] = seq_each_std
        seq_mean = seq_each_mean.mean()
        seq_std = seq_each_std.mean()
        feature_seq_means[group+"_log1p"] = seq_mean
        feature_seq_stds[group+"_log1p"] = seq_std

    seq_q_means = (feature_seq_means["state_q0001"] + feature_seq_means["state_q0002"] + feature_seq_means["state_q0003"]) / 3
    seq_q_stds = (feature_seq_stds["state_q0001"] + feature_seq_stds["state_q0002"] + feature_seq_stds["state_q0003"]) / 3
    feature_seq_means["state_q0001"] = seq_q_means
    feature_seq_stds["state_q0001"] = seq_q_stds
    feature_seq_means["state_q0002"] = seq_q_means
    feature_seq_stds["state_q0002"] = seq_q_stds
    feature_seq_means["state_q0003"] = seq_q_means
    feature_seq_stds["state_q0003"] = seq_q_stds

    seq_pbuf_means = (feature_seq_means["pbuf_ozone"] + feature_seq_means["pbuf_N2O"] + feature_seq_means["pbuf_CH4"]) / 3
    seq_pbuf_stds = (feature_seq_stds["pbuf_ozone"] + feature_seq_stds["pbuf_N2O"] + feature_seq_stds["pbuf_CH4"]) / 3
    feature_seq_means["pbuf_ozone"] = seq_pbuf_means
    feature_seq_stds["pbuf_ozone"] = seq_pbuf_stds
    feature_seq_means["pbuf_N2O"] = seq_pbuf_means
    feature_seq_stds["pbuf_N2O"] = seq_pbuf_stds
    feature_seq_means["pbuf_CH4"] = seq_pbuf_means
    feature_seq_stds["pbuf_CH4"] = seq_pbuf_stds

    # multiple weight at target
    for group_ix, group in enumerate(TARGET_SEQ_GROUPS):
        group_cols = [f"{group}_{i}" for i in range(60)]
        old_weight_arr = np.asarray([old_weight[c] for c in group_cols])
        train_seq_targets[:, group_ix, :] *= old_weight_arr
        valid_seq_targets[:, group_ix, :] *= old_weight_arr
        if "q000" in group:
            train_seq_targets[:, group_ix, :] *= 1e-30
            valid_seq_targets[:, group_ix, :] *= 1e-30

    # multiple weight at target
    before_shape = train_scaler_targets.shape
    old_weight_arr = np.asarray([old_weight[c] for c in TARGET_SCALER_COLS])
    train_scaler_targets = train_scaler_targets * old_weight_arr
    valid_scaler_targets = valid_scaler_targets * old_weight_arr
    assert train_scaler_targets.shape == before_shape

    return (train_seq_features, valid_seq_features, test_seq_features, train_scaler_features, valid_scaler_features, test_scaler_features,
            train_seq_targets, valid_seq_targets, train_scaler_targets, valid_scaler_targets,
            feature_seq_means, feature_seq_stds, feature_seq_each_means, feature_seq_each_stds)



def calc_normalize_stats(train_seq_features, train_scaler_features):
   
    # normalize scaler features
    feature_scaler_mean = train_scaler_features.mean(axis=0)
    feature_scaler_std = train_scaler_features.std(axis=0)
    feature_scaler_std = np.maximum(feature_scaler_std, 1e-8)

    # normalize seq features
    feature_seq_means = {}
    feature_seq_stds = {}
    feature_seq_each_means = {}
    feature_seq_each_stds = {}
    for group_ix, group in enumerate(FEATURE_SEQ_GROUPS):
        seq_each_mean = train_seq_features[:, group_ix, :].mean(axis=0)
        seq_each_std = train_seq_features[:, group_ix, :].std(axis=0)
        seq_each_std = np.maximum(seq_each_std, 1e-8)
        feature_seq_each_means[group] = seq_each_mean
        feature_seq_each_stds[group] = seq_each_std
        seq_mean = seq_each_mean.mean()
        seq_std = seq_each_std.mean()
        feature_seq_means[group] = seq_mean
        feature_seq_stds[group] = seq_std

    for group_ix, group in enumerate(FEATURE_SEQ_GROUPS):
        if not "state_q000" in group:
            continue
        seq_each_mean = np.log1p(train_seq_features[:, group_ix, :]).mean(axis=0)
        seq_each_std = np.log1p(train_seq_features[:, group_ix, :]).std(axis=0)
        seq_each_std = np.maximum(seq_each_std, 1e-8)
        feature_seq_each_means[group+"_log1p"] = seq_each_mean
        feature_seq_each_stds[group+"_log1p"] = seq_each_std
        seq_mean = seq_each_mean.mean()
        seq_std = seq_each_std.mean()
        feature_seq_means[group+"_log1p"] = seq_mean
        feature_seq_stds[group+"_log1p"] = seq_std

    seq_q_means = (feature_seq_means["state_q0001"] + feature_seq_means["state_q0002"] + feature_seq_means["state_q0003"]) / 3
    seq_q_stds = (feature_seq_stds["state_q0001"] + feature_seq_stds["state_q0002"] + feature_seq_stds["state_q0003"]) / 3
    feature_seq_means["state_q0001"] = seq_q_means
    feature_seq_stds["state_q0001"] = seq_q_stds
    feature_seq_means["state_q0002"] = seq_q_means
    feature_seq_stds["state_q0002"] = seq_q_stds
    feature_seq_means["state_q0003"] = seq_q_means
    feature_seq_stds["state_q0003"] = seq_q_stds

    seq_pbuf_means = (feature_seq_means["pbuf_ozone"] + feature_seq_means["pbuf_N2O"] + feature_seq_means["pbuf_CH4"]) / 3
    seq_pbuf_stds = (feature_seq_stds["pbuf_ozone"] + feature_seq_stds["pbuf_N2O"] + feature_seq_stds["pbuf_CH4"]) / 3
    feature_seq_means["pbuf_ozone"] = seq_pbuf_means
    feature_seq_stds["pbuf_ozone"] = seq_pbuf_stds
    feature_seq_means["pbuf_N2O"] = seq_pbuf_means
    feature_seq_stds["pbuf_N2O"] = seq_pbuf_stds
    feature_seq_means["pbuf_CH4"] = seq_pbuf_means
    feature_seq_stds["pbuf_CH4"] = seq_pbuf_stds

    return feature_scaler_mean, feature_scaler_std, feature_seq_means, feature_seq_stds, feature_seq_each_means, feature_seq_each_stds


class Ptrans:
    def __init__(self, mode="npy"):# , hyai, hybi, p0):
        
        hyai = np.array([5.58810705e-05, 1.00814552e-04, 1.81402085e-04, 3.24444509e-04,
            5.74056761e-04, 9.98635562e-04, 1.69607596e-03, 2.79347861e-03,
            4.43938435e-03, 6.79228850e-03, 1.00142179e-02, 1.42747608e-02,
            1.97588953e-02, 2.66627009e-02, 3.51659916e-02, 4.53891697e-02,
            5.73600949e-02, 7.10183619e-02, 8.62609533e-02, 1.02999231e-01,
            1.21183316e-01, 1.40772291e-01, 1.61670345e-01, 1.81999009e-01,
            1.76911184e-01, 1.71712893e-01, 1.66457312e-01, 1.61163736e-01,
            1.55816446e-01, 1.50377526e-01, 1.44804991e-01, 1.39066613e-01,
            1.33144781e-01, 1.27034194e-01, 1.20738293e-01, 1.14269959e-01,
            1.07657953e-01, 1.00955234e-01, 9.42421137e-02, 8.76184241e-02,
            8.11845793e-02, 7.50186097e-02, 6.91601845e-02, 6.36099509e-02,
            5.83442610e-02, 5.33368416e-02, 4.85756800e-02, 4.40670125e-02,
            3.98260499e-02, 3.58611336e-02, 3.21605828e-02, 2.86887250e-02,
            2.53918368e-02, 2.22097376e-02, 1.90871734e-02, 1.59808815e-02,
            1.28613784e-02, 9.71093390e-03, 6.51999580e-03, 3.28380931e-03,
            0.00000000e+00])
        hybi = np.array([0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.00167846, 0.02958677,
            0.05810102, 0.08692953, 0.11596645, 0.14529801, 0.17513219,
            0.20569929, 0.23717609, 0.26965919, 0.30317767, 0.33771266,
            0.3731935 , 0.40946242, 0.44622894, 0.4830525 , 0.51938551,
            0.55467717, 0.58849944, 0.62063474, 0.6510795 , 0.67996346,
            0.70743073, 0.73354719, 0.75827864, 0.78154165, 0.80329046,
            0.82358913, 0.84263336, 0.86071783, 0.87817264, 0.89530088,
            0.91233986, 0.92945131, 0.94673249, 0.96423578, 0.98198728,
            1.        ])

        p0 = 100000

        self.hyai = hyai[np.newaxis, :] # [1, 61]
        self.hybi = hybi[np.newaxis, :] # [1, 61]
        self.p0 = p0 # scalar
        self.mode = mode

    def transform_npy(self, ps):
        ps = ps.reshape(-1,1)
        pint = self.p0 * self.hyai + ps * self.hybi
        dp = pint[:,1:] - pint[:,:-1]
        p = (pint[:,1:] + pint[:,:-1])/2
        return p, dp, pint[:,1:], pint[:,:-1]

    def transform(self, ps): # ps is [batch]
        return self.transform_npy(ps)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            self = pickle.load(f)
        return self     


def saturation_vapor_pressure(temperature):
    """
    input is kelvin temperature
    飽和蒸気圧[Pa]
    飽和水蒸気量[kg/m3]
    tetens
    
    """
    # t = df["air_temperature"].values
    # saturation_vapor_p = 6.11 * (10**(7.5*(temperature - 273.15)/(temperature - 35.85)+2)) # hpa->pa
    x = 1 - temperature / 647.3
    # ワグナ
    saturation_vapor_p = 22120000 * np.exp((-7.76451 * x + 1.45838 * x**1.5 - 2.7758 * x**3 - 1.23303 * x**6) / (1 - x))
    saturation_vapor_m = 217 * saturation_vapor_p / (temperature * 1e6)
    return saturation_vapor_p, saturation_vapor_m

def dew_temperature(vapor_pressure):
    """
    tetens
    """
    y = np.log(vapor_pressure / 611.153)
    cel_temperature = 12.1197 * y + (5.25112/10) * y**2 + (1.92206/100) * y**3 + (3.8403/10000) * y**4
    # cel_temperature = 237.3 * np.log10(610.78 / vapor_pressure) / (np.log10(vapor_pressure / 610.78) - 7.5)
    temperature = cel_temperature + 273.15
    return temperature

def vapor_pressure(specific_humidity, pressure):
    """
    specific = 0.622 * pv / (p - 0.378 * pv)
    pv = spe * p / (0.622 + 0.378 * spe)        
    """
    q_scale_adjust = 1.0
    pv = pressure * (specific_humidity / (0.622 * q_scale_adjust + 0.378 * specific_humidity))
    return pv


def add_vapor_pressure(df):
    p, dp, p_u, p_l = Ptrans(mode="npy").transform(df["state_ps"].to_numpy())
    del p_u, p_l, dp
    gc.collect()

    vapor_pressure_ = vapor_pressure(df.select([f"state_q0001_{i}" for i in range(60)]).to_numpy() * 1e-30, p)
    saturation_vapor_p, saturation_vapor_m = saturation_vapor_pressure(df.select([f"state_t_{i}" for i in range(60)]).to_numpy())
    rel_vapor_pressure = vapor_pressure_ / saturation_vapor_p
    rel_vaper_pressure_cols = [f"rel_vapor_pressure_{i}" for i in range(60)]
    df = df.with_columns(
        pl.DataFrame(rel_vapor_pressure, schema=rel_vaper_pressure_cols),
    )
    del p, saturation_vapor_p, saturation_vapor_m, rel_vapor_pressure
    dew_point = dew_temperature(vapor_pressure_)
    diff_dew_point = df.select([f"state_t_{i}" for i in range(60)]).to_numpy() - dew_point
    diff_dew_point_cols = [f"diff_dew_point_{i}" for i in range(60)]
    df = df.with_columns(
        pl.DataFrame(diff_dew_point, schema=diff_dew_point_cols),
    )

    gc.collect()
    return df


class LeapDataset(Dataset):
    def __init__(self, seq_features: np.ndarray, scaler_features: np.ndarray,
                 seq_targets: Union[np.ndarray, None] = None, scaler_targets: Union[np.ndarray, None] = None):
        self.scaler_features = scaler_features
        self.seq_features = seq_features

        if seq_targets is not None:
            self.scaler_targets = scaler_targets
            self.seq_targets = seq_targets.transpose(0, 2, 1)  # (N, 6, 60) -> (N, 60, 6)
        else:
            self.seq_targets = None
            self.scaler_targets = None

    def __len__(self):
        return len(self.scaler_features)
    
    def __getitem__(self, ix):
        seq_features = self.seq_features[ix]
        scaler_features = self.scaler_features[ix]
        # add log1p at state_q000
        seq_features_log1p = np.log1p(seq_features[0:3])
        seq_features = np.concatenate([seq_features, seq_features_log1p], axis=0)
        seq_features_each_normed = np.zeros_like(seq_features)
        seq_features_group_normed = np.zeros_like(seq_features)
        for group_ix, group in enumerate(FEATURE_SEQ_GROUPS + [f"{group}_log1p" for group in FEATURE_SEQ_GROUPS if "state_q000" in group]):
            seq_features_each_normed[group_ix] = (seq_features[group_ix] - feature_seq_each_means[group]) / feature_seq_each_stds[group]
            # seq_features_group_normed[group_ix] = (seq_features[group_ix] - feature_seq_means[group]) / feature_seq_stds[group]
        # seq_features = np.concatenate([seq_features_each_normed, seq_features_group_normed], axis=0)
        seq_features = seq_features_each_normed

        if self.scaler_targets is not None:
            scaler_targets = self.scaler_targets[ix]  # (8,)
            seq_targets = self.seq_targets[ix]  # (60, 6)
            seq_targets_diff = np.diff(seq_targets, axis=0)  # (59, 6)
            seq_targets_diff = np.concatenate([np.zeros((1, 6)), seq_targets_diff], axis=0)  # (60, 6)
            return {
                "scaler_features": torch.tensor(scaler_features, dtype=torch.float),
                "seq_features": torch.tensor(seq_features, dtype=torch.float),
                "scaler_targets": torch.tensor(scaler_targets, dtype=torch.float),
                "seq_targets": torch.tensor(seq_targets, dtype=torch.float),
                "seq_targets_diff": torch.tensor(seq_targets_diff, dtype=torch.float)
            }
        else:
            return {
                "scaler_features": torch.tensor(scaler_features, dtype=torch.float),
                "seq_features": torch.tensor(seq_features, dtype=torch.float),
            }


# pbuf_ozone 60
# state_t 60
# state_u 60
# state_q0002 60
# state_v 60
# state_q0001 60
# pbuf_N2O 60
# pbuf_CH4 60
# state_q0003 60

class FeatureScale(nn.Module):
    def __init__(self, input_dim):
        super(FeatureScale, self).__init__()
        self.weights = nn.Parameter(torch.ones(input_dim))
        self.biases = nn.Parameter(torch.zeros(input_dim))
        
    def forward(self, x):
        # 各次元ごとに対応するパラメータを掛ける
        return x * self.weights + self.biases


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_features),
            # nn.ReLU(),
            nn.GELU(),
            nn.Conv1d(out_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_features),
            # nn.ReLU()
            nn.GELU()
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(out_features, nhead=8, dim_feedforward=out_features, dropout=0.0, batch_first=True),
            num_layers=1,
        )


    def forward(self, x):
        out = self.conv(x)
        out = out + x
        out = out.transpose(1, 2)
        out = self.transformer(out)
        out = out.transpose(1, 2)
        return out


class LeapModel(nn.Module):
    def __init__(self):
        super(LeapModel, self).__init__()
        # 60 sequences 1d cnn
        self.feature_scale = nn.ModuleList([
            FeatureScale(60) for _ in range((len(FEATURE_SEQ_GROUPS)+3) * 3 * 1)
        ])
        self.positional_encoding = nn.Embedding(60, 128)
        self.input_linear = nn.Linear((len(FEATURE_SEQ_GROUPS)+3) * 3 * 1, 128)  # current, diff
        n_hidden = 128
        self.other_feats_mlp = nn.Sequential(
            nn.Linear(len(FEATURE_SCALER_COLS), n_hidden),
            nn.BatchNorm1d(n_hidden),
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            # nn.ReLU(),
            nn.GELU()
        )
        self.other_feats_proj = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for _ in range(60)])
        # layer norm
        layer_norm_in = n_hidden + 128
        self.seq_layer_norm = nn.LayerNorm(layer_norm_in)
        # cnn
        n_cnn_hidden = n_hidden + 128
        self.cnn1 = nn.Sequential(
            ResidualBlock(n_cnn_hidden, n_cnn_hidden, 5, 1, 2),
            ResidualBlock(n_cnn_hidden, n_cnn_hidden, 5, 1, 2),
            ResidualBlock(n_cnn_hidden, n_cnn_hidden, 5, 1, 2),
            ResidualBlock(n_cnn_hidden, n_cnn_hidden, 5, 1, 2)
        )
        # lstm
        self.lstm = nn.Sequential(
            nn.LSTM(n_cnn_hidden, n_cnn_hidden, 2, batch_first=True, bidirectional=True, dropout=0.0),
        )
        # output layer
        output_seq_mlp_input_dim = n_cnn_hidden*2
        self.seq_output_mlp = nn.Sequential(
            nn.Linear(output_seq_mlp_input_dim, n_hidden*2),
            nn.GELU(),
            nn.Linear(n_hidden*2, n_hidden*2),
            nn.GELU(),
            nn.Linear(n_hidden*2, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, len(TARGET_SEQ_GROUPS) * 2)
        )
        output_scaler_mlp_input_dim = n_cnn_hidden*2*60
        self.scaler_layer_norm = nn.LayerNorm(output_scaler_mlp_input_dim)
        self.scaler_output_mlp = nn.Sequential(
            nn.Linear(output_scaler_mlp_input_dim, n_hidden),
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(n_hidden, n_hidden),
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(n_hidden, len(TARGET_SCALER_COLS))
        )

    def forward(self, scaler_features, seq_features):
        # concat dim60 cols
        dim60_x = []
        scale_ix = 0
        # for group_ix in range(len(FEATURE_SEQ_GROUPS) * 2):
        for group_ix in range((len(FEATURE_SEQ_GROUPS)+3) * 1):
            origin_x = seq_features[:, group_ix, :]
            x = self.feature_scale[scale_ix](origin_x)  # (batch, 60)
            scale_ix += 1
            x = x.unsqueeze(-1)  # (batch, 60, 1)
            dim60_x.append(x)
            # diff feature
            x_diff = origin_x[:, 1:] - origin_x[:, :-1]  # (batch, 59)
            x_diff = torch.cat([origin_x.new_zeros(origin_x.size(0), 1), x_diff], dim=1)  # (batch, 60)
            x_diff = self.feature_scale[scale_ix](x_diff)  # (batch, 60)
            scale_ix += 1
            x_diff = x_diff.unsqueeze(-1)  # (batch, 60, 1)
            dim60_x.append(x_diff)
            # diff diff feature
            x_diff = origin_x[:, 1:] - origin_x[:, :-1]
            x_diff_diff = x_diff[:, 1:] - x_diff[:, :-1]  # (batch, 58)
            x_diff_diff = torch.cat([origin_x.new_zeros(origin_x.size(0), 2), x_diff_diff], dim=1)  # (batch, 60)
            x_diff_diff = self.feature_scale[scale_ix](x_diff_diff)  # (batch, 60)
            scale_ix += 1
            x_diff_diff = x_diff_diff.unsqueeze(-1)  # (batch, 60, 1)
            dim60_x.append(x_diff_diff)

        x = torch.cat(dim60_x, dim=2)  # (batch, 60, M)
        position = torch.arange(0, 60, device=x.device).unsqueeze(0).repeat(x.size(0), 1)  # (batch, 60)
        position = self.positional_encoding(position)  # (batch, 60, 16)
        x = self.input_linear(x)  # (batch, seq_len, 128)
        x = x + position
        # other cols
        scaler_x = scaler_features  # (batch, n_feats)
        scaler_x = self.other_feats_mlp(scaler_x)  # (batch, hidden)
        scaler_x_list = []
        for i in range(60):
            scaler_x_list.append(self.other_feats_proj[i](scaler_x))
        scaler_x = torch.stack(scaler_x_list, dim=1)  # (batch, 60, hidden)
        # repeat to match seq_len
        # scaler_x = scaler_x.unsqueeze(1).repeat(1, x.size(1), 1)  # (batch, seq_len, hidden)
        # concat
        x = torch.cat([x, scaler_x], dim=2)  # (batch, seq_len, hidden*2)
        x = self.seq_layer_norm(x)

        x = x.transpose(1, 2)  # (batch, hidden, seq_len)
        x = self.cnn1(x)  # (batch, hidden, seq_len)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)  # (batch, seq_len, hidden)

        # seq_head
        seq_output = self.seq_output_mlp(x)  # (batch, seq_len * 2, n_targets)
        seq_diff_output = seq_output[:, :, len(TARGET_SEQ_GROUPS):]
        seq_output = seq_output[:, :, :len(TARGET_SEQ_GROUPS)]
        # scaler_head
        scaler_x = x.reshape(x.size(0), -1)
        scaler_x = self.scaler_layer_norm(scaler_x)
        scaler_output = self.scaler_output_mlp(scaler_x)
        return seq_output, scaler_output, seq_diff_output


def train_one_epoch(model, loss_fn, data_loader, optimizer,
                    device, scheduler, epoch, scaler=None, awp=None, ema=None):
    # get batch data loop
    epoch_loss = 0
    epoch_data_num = len(data_loader.dataset)

    model.train()

    bar = tqdm(enumerate(data_loader), total=len(data_loader))

    scaler_weight_arr = np.asarray([new_weight[c] for c in TARGET_SCALER_COLS])
    seq_weight_arr = np.asarray([[new_weight[f"{c}_{i}"] for c in TARGET_SEQ_GROUPS] for i in range(60)])
    scaler_weight_mask = np.where(scaler_weight_arr == 0, 0, 1)
    seq_weight_mask = np.where(seq_weight_arr == 0, 0, 1)
    seq_weight_mask[12:27, TARGET_SEQ_GROUPS.index("ptend_q0002")] = 0.0
    scaler_weight_mask = torch.tensor(scaler_weight_mask, dtype=torch.float).to(device)
    seq_weight_mask = torch.tensor(seq_weight_mask, dtype=torch.float).to(device)

    for iter_i, batch in bar:
        # input
        seq_features = batch["seq_features"].to(device)
        scaler_features = batch["scaler_features"].to(device)
        seq_targets = batch["seq_targets"].to(device)
        scaler_targets = batch["scaler_targets"].to(device)
        seq_targets_diff = batch["seq_targets_diff"].to(device)
        batch_size = len(scaler_targets)

        # zero grad
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            with amp.autocast(enabled=CFG.use_fp16):
                seq_preds, scaler_preds, seq_diff_preds = model(scaler_features, seq_features)
                # mask
                seq_preds = seq_preds * seq_weight_mask
                scaler_preds = scaler_preds * scaler_weight_mask
                seq_diff_preds = seq_diff_preds * seq_weight_mask
                scaler_targets = scaler_targets * scaler_weight_mask
                seq_targets = seq_targets * seq_weight_mask
                seq_targets_diff = seq_targets_diff * seq_weight_mask
                # loss function
                seq_loss = loss_fn(seq_preds.reshape(-1), seq_targets.reshape(-1))
                scaler_loss = loss_fn(scaler_preds.reshape(-1), scaler_targets.reshape(-1))
                seq_diff_loss = loss_fn(seq_diff_preds.reshape(-1), seq_targets_diff.reshape(-1))
                loss = seq_loss + scaler_loss + seq_diff_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            epoch_loss += loss.item()
        
        bar.set_postfix(OrderedDict(loss=loss.item(), lr=optimizer.param_groups[0]['lr']))
    
    epoch_loss_per_data = epoch_loss / epoch_data_num
    return epoch_loss_per_data


def valid_one_epoch(model, loss_fn, data_loader, device):
    # get batch data loop
    epoch_loss = 0
    epoch_data_num = len(data_loader.dataset)
    seq_pred_list = []
    scaler_pred_list = []
    seq_target_list = []
    scaler_target_list = []
    bar = tqdm(enumerate(data_loader), total=len(data_loader))

    model.eval()
    for iter_i, batch in bar:
        # input
        seq_features = batch["seq_features"].to(device)
        scaler_features = batch["scaler_features"].to(device)
        seq_targets = batch["seq_targets"].to(device)
        scaler_targets = batch["scaler_targets"].to(device)
        seq_targets_diff = batch["seq_targets_diff"].to(device)
        batch_size = len(scaler_targets)

        with torch.no_grad():
            seq_preds, scaler_preds, seq_diff_preds = model(scaler_features, seq_features)
            # seq_preds: (batch, seq_len, n_targets)
            # scaler_preds: (batch, n_targets)
            seq_loss = loss_fn(seq_preds.reshape(-1), seq_targets.reshape(-1))
            scaler_loss = loss_fn(scaler_preds.reshape(-1), scaler_targets.reshape(-1))
            seq_diff_loss = loss_fn(seq_diff_preds.reshape(-1), seq_targets_diff.reshape(-1))
            loss = seq_loss + scaler_loss + seq_diff_loss
            epoch_loss += loss.item()

        seq_pred_list.append(seq_preds.detach().cpu().numpy())  # (n_data, seq_len, n_targets)
        scaler_pred_list.append(scaler_preds.detach().cpu().numpy())
        seq_target_list.append(seq_targets.detach().cpu().numpy())  # (n_data, seq_len, n_targets)
        scaler_target_list.append(scaler_targets.detach().cpu().numpy())

    seq_val_preds = np.concatenate(seq_pred_list)  # (n_data, seq_len, n_targets)
    seq_val_targets = np.concatenate(seq_target_list)  # (n_data, seq_len, n_targets)
    scaler_val_preds = np.concatenate(scaler_pred_list)  # (n_data, n_targets)
    scaler_val_targets = np.concatenate(scaler_target_list)  # (n_data, n_targets)
    epoch_loss_per_data = epoch_loss / epoch_data_num
    return epoch_loss_per_data, seq_val_preds, seq_val_targets, scaler_val_preds, scaler_val_targets


def train_one_epoch_all_data(model, loss_fn, data_loader, optimizer,
                    device, scheduler, epoch, norm_dict, scaler=None, awp=None, ema=None):
    # get batch data loop
    epoch_loss = 0
    epoch_data_num = len(data_loader.dataset)

    model.train()

    bar = tqdm(enumerate(data_loader), total=len(data_loader))

    scaler_weight_arr = np.asarray([new_weight[c] for c in TARGET_SCALER_COLS])
    seq_weight_arr = np.asarray([[new_weight[f"{c}_{i}"] for c in TARGET_SEQ_GROUPS] for i in range(60)])
    scaler_weight_mask = np.where(scaler_weight_arr == 0, 0, 1)
    seq_weight_mask = np.where(seq_weight_arr == 0, 0, 1)
    seq_weight_mask[12:27, TARGET_SEQ_GROUPS.index("ptend_q0002")] = 0.0
    scaler_weight_mask = torch.tensor(scaler_weight_mask, dtype=torch.float).to(device)
    seq_weight_mask = torch.tensor(seq_weight_mask, dtype=torch.float).to(device)

    # change to numpy
    all_feature_seq_groups = FEATURE_SEQ_GROUPS + [f"{group}_log1p" for group in FEATURE_SEQ_GROUPS if "state_q000" in group]
    scaler_mean_arr = torch.from_numpy(norm_dict["feature_scaler_mean"]).float().to(device)
    scaler_std_arr = torch.from_numpy(norm_dict["feature_scaler_std"]).float().to(device)
    seq_means_arr = torch.from_numpy(np.asarray([norm_dict["feature_seq_means"][c] for c in all_feature_seq_groups])).float().to(device)
    seq_stds_arr = torch.from_numpy(np.asarray([norm_dict["feature_seq_stds"][c] for c in all_feature_seq_groups])).float().to(device)
    seq_each_means_arr = torch.from_numpy(np.asarray([norm_dict["feature_seq_each_means"][c] for c in all_feature_seq_groups])).float().to(device)
    seq_each_stds_arr = torch.from_numpy(np.asarray([norm_dict["feature_seq_each_stds"][c] for c in all_feature_seq_groups])).float().to(device)

    for iter_i, batch in bar:
        # input
        seq_features = batch["seq_features.npy"].to(device)
        scaler_features = batch["scaler_features.npy"].to(device)
        seq_targets = batch["seq_target.npy"].to(device)
        scaler_targets = batch["scaler_target.npy"].to(device)

        seq_features_log1p = torch.log1p(seq_features[:, 0:3, :])
        seq_features = torch.cat([seq_features, seq_features_log1p], axis=1)

        seq_targets = seq_targets.transpose(1, 2)
        seq_targets_diff = torch.diff(seq_targets, dim=1)
        seq_targets_diff = torch.cat([torch.zeros_like(seq_targets_diff[:, :1], device=device), seq_targets_diff], dim=1)
        # normalize feature
        seq_features = (seq_features - seq_each_means_arr) / seq_each_stds_arr
        scaler_features = (scaler_features - scaler_mean_arr) / scaler_std_arr


        # zero grad
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            with amp.autocast(enabled=CFG.use_fp16):
                seq_preds, scaler_preds, seq_diff_preds = model(scaler_features, seq_features)
                # mask
                seq_preds = seq_preds * seq_weight_mask
                scaler_preds = scaler_preds * scaler_weight_mask
                seq_diff_preds = seq_diff_preds * seq_weight_mask
                scaler_targets = scaler_targets * scaler_weight_mask
                seq_targets = seq_targets * seq_weight_mask
                seq_targets_diff = seq_targets_diff * seq_weight_mask
                # loss function
                seq_loss = loss_fn(seq_preds.reshape(-1), seq_targets.reshape(-1))
                scaler_loss = loss_fn(scaler_preds.reshape(-1), scaler_targets.reshape(-1))
                seq_diff_loss = loss_fn(seq_diff_preds.reshape(-1), seq_targets_diff.reshape(-1))
                loss = seq_loss + scaler_loss + seq_diff_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            epoch_loss += loss.item()
        
        bar.set_postfix(OrderedDict(loss=loss.item(), lr=optimizer.param_groups[0]['lr']))

    epoch_loss_per_data = epoch_loss / epoch_data_num
    return epoch_loss_per_data




def calc_metric(seq_val_preds, seq_val_targets, scaler_val_preds, scaler_val_targets, per_class: bool) -> float:
    r2_scores = []
    r2_scores_dict = {}
    for i, t_col in tqdm(enumerate(TARGET_SCALER_COLS)):
        if new_weight[t_col] == 0:
            r2_scores.append(1)

        if old_weight[t_col] != 0:
            _targets = scaler_val_targets[:, i] / old_weight[t_col]
            _preds = scaler_val_preds[:, i] / old_weight[t_col]
        else:
            _targets = scaler_val_targets[:, i]
            _preds = scaler_val_preds[:, i]
        _r2_score = r2_score(_targets, _preds)
        if per_class:
            logger.info(f"{t_col}: {_r2_score}")
        if _r2_score >= 0:
            r2_scores.append(_r2_score)
        else:
            r2_scores.append(0)
        r2_scores_dict[t_col] = _r2_score

    for i, seq_group in enumerate(TARGET_SEQ_GROUPS):
        seq_target_cols = [f"{seq_group}_{ix}" for ix in range(60)]
        for j, t_col in enumerate(seq_target_cols):
            if new_weight[t_col] == 0:  # 0埋めするやつ
                _r2_score = 1
            elif t_col in [f"ptend_q0002_{i}" for i in range(27)]:  # -state/1200で埋めるやつ
                _r2_score = 1
            else:
                if old_weight[t_col] != 0:
                    _targets = seq_val_targets[:, j, i] / old_weight[t_col]
                    _preds = seq_val_preds[:, j, i] / old_weight[t_col]
                else:
                    _targets = seq_val_targets[:, j, i]
                    _preds = seq_val_preds[:, j, i]
                _r2_score = r2_score(_targets, _preds)
            if per_class:
                logger.info(f"{t_col}: {_r2_score}")
            if _r2_score >= 0:
                r2_scores.append(_r2_score)
            else:
                r2_scores.append(0)
            r2_scores_dict[t_col] = _r2_score

    cv_score = np.mean(r2_scores)
    return cv_score, r2_scores_dict


def train_run(train_seq_X=None, train_scaler_X=None, train_seq_y=None, train_scaler_y=None,
              valid_seq_X=None, valid_scaler_X=None, valid_seq_y=None, valid_scaler_y=None,
              feature_scaler_mean=None, feature_scaler_std=None, feature_seq_means=None, feature_seq_stds=None, feature_seq_each_means=None, feature_seq_each_stds=None,
              model_prefix=""):
    
    set_seed(CFG.seed)
    
    device = torch.device(
        f"cuda:{GPU_IX}" if torch.cuda.is_available() else "cpu"
    )
    # device = torch.device("cpu")
    logger.info(f"train run device : {device}")
    
    ###################################
    # Model and Tokenizer
    ###################################
    model = LeapModel()
    model.to(device)

    ema = ModelEmaV2(model, decay=0.9999, device=device)
    ema.set(model)

    scaler = amp.GradScaler(enabled=CFG.use_fp16)
    
    ###################################
    # Make data
    ###################################

    if not ALL_TRAIN:
        
        train_dataset = LeapDataset(train_seq_X, train_scaler_X, train_seq_y, train_scaler_y)
        logger.info("create train dataset")
        del train_scaler_X, train_scaler_y, train_seq_X, train_seq_y
        gc.collect()
        train_loader = DataLoader(
            train_dataset,
            batch_size=CFG.batch_size,
            num_workers=CFG.num_workers,
            shuffle=True,
            drop_last=True
        )

        if valid_seq_X is not None:
            valid_dataset = LeapDataset(valid_seq_X, valid_scaler_X, valid_seq_y, valid_scaler_y)
            valid_loader = DataLoader(
                valid_dataset, 
                batch_size=CFG.batch_size,
                num_workers=CFG.num_workers,
                shuffle=False,
            )
            del valid_scaler_X, valid_scaler_y, valid_seq_X, valid_seq_y
            gc.collect()
        else:
            valid_dataset = None
            valid_loader = None
    else:
        shard_pattern = '/data/hf_all/shards_01/shards-{00000..05406}.tar'

        train_dataset = wds.WebDataset(shard_pattern, shardshuffle=True)
        train_dataset = train_dataset.shuffle(size=CFG.batch_size * 8)
        train_dataset = train_dataset.with_length(60774252)
        train_dataset = train_dataset.decode('torch')
        # train_dataset = LeapDataset(train_seq_X, train_scaler_X, train_seq_y, train_scaler_y)
        print("create train dataset")
        gc.collect()
        train_loader = DataLoader(
            train_dataset,
            batch_size=CFG.batch_size,
            num_workers=CFG.num_workers,
            # shuffle=True,
            drop_last=True
        )

    ##################
    # Optimiizer
    ##################
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, betas=(0.9, 0.98))

    awp = None
   
    ##################
    # lr scheduler
    ##################
    num_train_optimization_steps = len(train_loader) * CFG.n_epoch
    num_warmup_steps = int(num_train_optimization_steps * CFG.num_warmup_steps_rate)

    # scheduler = transformers.get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=num_warmup_steps,
    #     num_training_steps=num_train_optimization_steps
    # )
    CFG.scheduler_params["T_max"] = num_train_optimization_steps
    scheduler = CFG.SchedulerClass(optimizer, **CFG.scheduler_params)

    ##################
    # loss function
    ##################
    # loss_fn = nn.MSELoss()
    loss_fn = nn.HuberLoss()

    ###############################
    # train epoch loop
    ###############################
    # iteration and loss count
    results_list = []
    val_preds_list = []
    logger.info("start train")
    for epoch in range(CFG.n_epoch):
        
        t_epoch_start = time.time()

        if not ALL_TRAIN:
            # train loop
            train_epoch_loss = train_one_epoch(
                model, loss_fn, train_loader, optimizer, device, scheduler,
                epoch=epoch, scaler=scaler, awp=awp, ema=ema
            )

            # valid loop
            if valid_loader is not None:
                valid_epoch_loss, seq_val_preds, seq_val_targets, scaler_val_preds, scaler_val_targets = valid_one_epoch(
                    ema.module, loss_fn, valid_loader, device
                )
                # calc metric
                if epoch == CFG.n_epoch - 1:
                    per_class = True
                else:
                    per_class = False
                val_score, val_score_dict = calc_metric(seq_val_preds, seq_val_targets, scaler_val_preds, scaler_val_targets, per_class=per_class)
                # val_score = 0
                results_list.append(val_score_dict)
                t_epoch_finish = time.time()
                elapsed_time = t_epoch_finish - t_epoch_start
            else:
                valid_epoch_loss = None
                val_score = None
                seq_val_preds = None
                scaler_val_preds = None

            # learning rate step
            lr = optimizer.param_groups[0]['lr']
            
            # save results
            results = {
                "epoch": epoch + 1,
                "lr": lr,
                "train_loss": train_epoch_loss,
                "valid_loss": valid_epoch_loss,
                "score": val_score
            }
            logger.info(results)

            if epoch == CFG.n_epoch -1:
                model_save_path = EXP_DIR / f"{model_prefix}last-checkpoint.bin"
                torch.save({
                    'model_state_dict': ema.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                }, model_save_path)
            

        else:
            norm_dict = {
                "feature_scaler_mean": feature_scaler_mean,
                "feature_scaler_std": feature_scaler_std,
                "feature_seq_means": feature_seq_means,
                "feature_seq_stds": feature_seq_stds,
                "feature_seq_each_means": feature_seq_each_means,
                "feature_seq_each_stds": feature_seq_each_stds
            }
            train_epoch_loss = train_one_epoch_all_data(
                model, loss_fn, train_loader, optimizer, device, scheduler, epoch=epoch, norm_dict=norm_dict,
                scaler=scaler, awp=awp, ema=ema
            )
            # learning rate step
            lr = optimizer.param_groups[0]['lr']
            
            # save results
            results = {
                "epoch": epoch + 1,
                "lr": lr,
                "train_loss": train_epoch_loss,
            }
            print(results)
            results_list.append(results)

            if epoch == CFG.n_epoch -1:
                model_save_path = EXP_DIR / f"{model_prefix}last-checkpoint.bin"
                torch.save({
                    'model_state_dict': ema.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                }, model_save_path)
            else:
                model_save_path = EXP_DIR / f"{model_prefix}-epoch{epoch}-checkpoint.bin"
                torch.save({
                    'model_state_dict': ema.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                }, model_save_path)
    if not ALL_TRAIN:
        return val_score, results_list, seq_val_preds, scaler_val_preds, seq_val_targets, scaler_val_targets
    else:
        return results_list



def fold_train(seed, fold_ix: int = 0):
    global train_seq_features, train_scaler_features, train_seq_targets, train_scaler_targets
    global valid_seq_features, valid_scaler_features, valid_seq_targets, valid_scaler_targets
    CFG.seed = seed
    val_scores = []

    if LIGHT_VERSION:
        num_train = int(len(train_seq_features) * 0.1)
        train_seq_features = train_seq_features[:num_train]
        train_scaler_features = train_scaler_features[:num_train]
        train_seq_targets = train_seq_targets[:num_train]
        train_scaler_targets = train_scaler_targets[:num_train]

    logger.info("="*30)
    logger.info(f"Fold{fold_ix}")
    logger.info("="*30)
    logger.info("training run start")

    if not ALL_TRAIN:
        val_score, score_list, seq_val_preds, scaler_val_preds, seq_val_targets, scaler_val_targets = train_run(
            train_seq_X=train_seq_features,
            train_scaler_X=train_scaler_features,
            train_seq_y=train_seq_targets,
            train_scaler_y=train_scaler_targets,
            valid_seq_X=valid_seq_features,
            valid_scaler_X=valid_scaler_features,
            valid_seq_y=valid_seq_targets,
            valid_scaler_y=valid_scaler_targets,
            model_prefix=f"fold{fold_ix}",
        )
        val_scores.append(val_score)

        logger.info(f"CV Score: {val_score}")
        return val_score, seq_val_preds, scaler_val_preds, seq_val_targets, scaler_val_targets, score_list
    else:
        score_list = train_run(
            feature_scaler_mean=feature_scaler_mean,
            feature_scaler_std=feature_scaler_std,
            feature_seq_means=feature_seq_means,
            feature_seq_stds=feature_seq_stds,
            feature_seq_each_means=feature_seq_each_means,
            feature_seq_each_stds=feature_seq_each_stds,
            model_prefix=f"fold{fold_ix}",
        )
        return score_list


if __name__ == "__main__":

    ALL_TRAIN = True

    train_df, test_df, old_weight, new_weight = load_data()
    valid_num = 0
    val_sprit = np.asarray([False] * len(train_df))
   
    FEATURE_ALL_COLS = train_df.columns[1:557]
    TARGET_ALL_COLS = train_df.columns[557:]

    train_df = add_vapor_pressure(train_df)
    test_df = add_vapor_pressure(test_df)
    # add_seq_features
    train_seq_features, test_seq_features, train_df, test_df = add_seq_features(train_df, test_df, val_sprit, valid_num, all_train=True)
    # add_scaler_features
    train_scaler_features, test_scaler_features, train_df, test_df = add_scaler_features(train_df, test_df, val_sprit, valid_num, all_train=True)
    del train_df, test_df
    gc.collect()
    (feature_scaler_mean, feature_scaler_std, feature_seq_means, feature_seq_stds, feature_seq_each_means, feature_seq_each_stds
        ) = calc_normalize_stats(train_seq_features, train_scaler_features)
    del train_seq_features, train_scaler_features
    gc.collect()

    score_list = fold_train(seed=2024, fold_ix=0)
