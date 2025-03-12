import gc
import json
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
import webdataset as wds
from tqdm import tqdm


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


def load_and_create_arr(filepath):
    df = pl.read_ipc(filepath, memory_map=False)
    df = add_vapor_pressure(df)

    old_weight = pl.read_csv("./data/old_sample_submission.csv", n_rows=1).to_dict(as_series=False)
    old_weight = {k: v[0] for k, v in old_weight.items()}

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
    seq_features = np.zeros((len(df), len(FEATURE_SEQ_GROUPS), 60))
    for group_ix, group in enumerate(FEATURE_SEQ_GROUPS):
        cols = [f"{group}_{i}" for i in range(60)]
        seq_features[:, group_ix] += df.select(cols).to_numpy()
        df = df.drop(cols)

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
    scaler_features = df.select(FEATURE_SCALER_COLS).to_numpy()
    df.drop(FEATURE_SCALER_COLS)

    TARGET_SEQ_GROUPS = [
        'ptend_t',
        'ptend_u',
        'ptend_v',
        'ptend_q0001',
        'ptend_q0002',
        'ptend_q0003'
    ]
    seq_targets = np.zeros((len(df), len(TARGET_SEQ_GROUPS), 60))
    for group_ix, group in enumerate(TARGET_SEQ_GROUPS):
        cols = [f"{group}_{i}" for i in range(60)]
        seq_targets[:, group_ix, :] += df.select(cols).to_numpy()
        df = df.drop(cols)

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
    scaler_targets = df.select(TARGET_SCALER_COLS).to_numpy()

    # normlize target
    # multiple weight at target
    for group_ix, group in enumerate(TARGET_SEQ_GROUPS):
        group_cols = [f"{group}_{i}" for i in range(60)]
        old_weight_arr = np.asarray([old_weight[c] for c in group_cols])
        seq_targets[:, group_ix, :] *= old_weight_arr
        if "q000" in group:
            seq_targets[:, group_ix, :] *= 1e-30

    # multiple weight at target
    before_shape = scaler_targets.shape
    old_weight_arr = np.asarray([old_weight[c] for c in TARGET_SCALER_COLS])
    scaler_targets = scaler_targets * old_weight_arr
    assert scaler_targets.shape == before_shape

    seq_features = seq_features.astype(np.float32)
    scaler_features = scaler_features.astype(np.float32)
    seq_targets = seq_targets.astype(np.float32)
    scaler_targets = scaler_targets.astype(np.float32)
    return seq_features, scaler_features, seq_targets, scaler_targets


def make_shard(filepath_list):

    shard_dir_path = Path(DATA_DIR / "shards_01")
    shard_dir_path.mkdir(exist_ok=True)
    shard_filename = str(shard_dir_path / 'shards-%05d.tar')

    shard_size = int(50 * 1000**2)  # 50MB each
    
    with wds.ShardWriter(
        shard_filename,
        maxsize=shard_size,
        ) as sink, tqdm(
            filepath_list
        ) as pbar:
        total_ix = 0
        for filepath in pbar:

            seq_features, scaler_features, seq_targets, scaler_targets = load_and_create_arr(filepath)

            for ix in range(len(seq_features)):

                sink.write({
                    "__key__": str(ix+total_ix),
                    "seq_features.npy": seq_features[ix],
                    "scaler_features.npy": scaler_features[ix],
                    "seq_target.npy": seq_targets[ix],
                    "scaler_target.npy": scaler_targets[ix],
                })
            total_ix += len(seq_features)

    dataset_size = total_ix

    dataset_size_filename = str(
        shard_dir_path / 'dataset-size.json')
    with open(dataset_size_filename, 'w') as fp:
        json.dump({
            "dataset size": dataset_size,
        }, fp)


if __name__ == "__main__":

    DATA_DIR = "../data/"

    filepath_list = list(Path(DATA_DIR).glob("*feather"))

    make_shard(filepath_list)
