import gc
import os
import polars as pl

INPUT_DIR = "/data/input/leap-atmospheric-physics-ai-climsim"
PAO_DIR = "/data/input/pao"
os.makedirs(PAO_DIR, exist_ok=True)

if __name__ == "__main__":

    weight = pl.read_csv(f"{INPUT_DIR}/new_sample_submission.csv", n_rows=1).to_dict(
        as_series=False
    )
    weight = {k: v[0] for k, v in weight.items()}
    train_df = pl.read_csv(f"{INPUT_DIR}/train.csv")
    print(train_df.shape)

    FEATURE_COLS = train_df.columns[1:557]
    TARGET_COLS = train_df.columns[557:]

    for col in FEATURE_COLS:
        if "q0001" in col or "q0002" in col or "q0003" in col:
            coef = 1e30
            train_df = train_df.with_columns(pl.col(col) * coef)
        train_df = train_df.with_columns(pl.col(col).cast(pl.Float32))
    for col in TARGET_COLS:
        if "q0001" in col or "q0002" in col or "q0003" in col:
            coef = 1e30
            train_df = train_df.with_columns(pl.col(col) * coef)
        train_df = train_df.with_columns(pl.col(col).cast(pl.Float32))

    train_df.write_parquet(f"{PAO_DIR}/new_train.parquet")
    del train_df
    gc.collect()
    print("Train Done")
    test_df = pl.read_csv(f"{INPUT_DIR}/test.csv")
    print(test_df.shape)
    for col in test_df.columns:
        if col == "sample_id":
            continue
        if "q0001" in col or "q0002" in col or "q0003" in col:
            coef = 1e30
            test_df = test_df.with_columns(pl.col(col) * coef)
        test_df = test_df.with_columns(pl.col(col).cast(pl.Float32))
    test_df.write_parquet(f"{PAO_DIR}/new_test.parquet")
    del test_df
    gc.collect()
    print("Test Done1")
    test_df = pl.read_csv(f"{INPUT_DIR}/test.csv")
    q_cols = [col for col in test_df.columns if "q000" in col]
    test_df = test_df.select(q_cols)
    test_df.write_parquet(f"{PAO_DIR}/new_test_original_q_fp64.parquet")
    print("Test Done2")
