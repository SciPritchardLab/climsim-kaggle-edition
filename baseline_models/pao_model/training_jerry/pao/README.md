## Pao Part

### 0. Prepare Row-Res data

This is created by K-mat
- Download from huggingface
- upscale state/ptend_q0001-3 cols (x 1e30)
- convert fp64 to fp32
- save feather

https://www.kaggle.com/code/kmat2019/leap-external-data

### 1. Preprocess and Convert Webdataset

I did some preprocess(including feature engineering)
I use webdataset for large data training, So I convert feather to webdataset.

At `1_1_lowres_preprocess_and_convert_to_wds.py`

- Feature engineering
  - My feature is just `vapor_pressure / saturation_vapor_pressure`.
- Preprocess
  - Multiple old sample_submission.csv weight to target
  - Reverse target upscaling of ptend_q0001-3 cols (x 1e-30)
  - Convert to 4 numpy array (scaler_feature, sequence_feature, scaler_targets, sequence_targets)

At `1_2_create_kaggle_dataset_parquet.py`

- Convert csv to parquet for smooth reading
- upscale state/ptend_q0001-3 cols (x 1e30)


```
python 1_1_lowres_preprocess_and_convert_to_wds.py
python 1_2_create_kaggle_dataset_parquet.py
```

### 2. Train & Inference(model: pao1)

```
python 2_1_pao1_train.py
python 2_2_pao1_inference.py
```

### 3. Train & Inference(model: pao2)

```
python 3_1_pao2_train.py
python 3_2_pao2_inference.py
```

### 4. Train & Inference(model: pao3)

```
python 4_1_pao3_train.py
python 4_2_pao3_inference.py
```


## Inference for new recreated test data


```
python 999_1_create_recreated_test_parquet.py
python 999_2_pao1_inference.py
python 999_3_pao2_inference.py
python 999_4_pao3_inference.py
```

