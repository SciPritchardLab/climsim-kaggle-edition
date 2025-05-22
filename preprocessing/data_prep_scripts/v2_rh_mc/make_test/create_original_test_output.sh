#!/bin/bash

cd ..
python process_data_v2.py \
            'E3SM-MMF.ml2steploc.0009-0[3456789]-*-*.nc' \
            'E3SM-MMF.ml2steploc.0009-1[012]-*-*.nc' \
            'E3SM-MMF.ml2steploc.0010-*-*-*.nc' \
            'E3SM-MMF.ml2steploc.0011-0[12]-*-*.nc' \
    --data_split 'test' \
    --stride_sample 12 \
    --start_idx 1 \
    --save_h5 True \
    --save_path '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/preprocessing/v2_rh_mc/test_set/original_test_vars/'