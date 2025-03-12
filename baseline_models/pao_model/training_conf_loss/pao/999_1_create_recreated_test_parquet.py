import gc
import polars as pl


if __name__ == "__main__":
    for version in ["01", "02", "03"]:
        print(f"Version: {version}")
        test_df = pl.read_csv(f"../recreated_{version}/test.csv")
        print(test_df.shape)
        for col in test_df.columns:
            if col == "sample_id":
                continue
            if "q0001" in col or "q0002" in col or "q0003" in col:
                coef = 1e+30
                test_df = test_df.with_columns(pl.col(col) * coef)
            test_df = test_df.with_columns(pl.col(col).cast(pl.Float32))
        test_df.write_parquet(f"../recreated_{version}/new_test.parquet")
        del test_df
        gc.collect()
        print("Test Done1")
        test_df = pl.read_csv(f"../recreated_{version}/test.csv")
        q_cols = [col for col in test_df.columns if "q000" in col]
        test_df = test_df.select(q_cols)
        test_df.write_parquet(f"../recreated_{version}/new_test_original_q_fp64.parquet")
        print("Test Done2")
