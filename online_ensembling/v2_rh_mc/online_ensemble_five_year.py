import os
import sys
import subprocess
import textwrap

# copy online_ensemble_template.sh to online_ensemble.sh
model_ensemble = 'five_year_runs'
num_months = 61
job_minutes_per_month = 10 # estimate of 10 minutes to simulate 1 month, can change for more expensive models
output_frequency = 'monthly'
ensemble_dir = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/online_runs/climsim3_ensembles_good/standard'
case_dir = f'{ensemble_dir}/{model_ensemble}'
compiled_esm_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/online_runs/climsim3_ensembles_good/empty_case/build/e3sm.exe'
email_address = 'jerryL9@uci.edu'

case_prefixes = ['unet_seed_7', \
                 'unet_seed_43', \
                 'unet_seed_1024', \
                 'squeezeformer_seed_7', \
                 'squeezeformer_seed_43', \
                 'squeezeformer_seed_1024', \
                 'pure_resLSTM_seed_7', \
                 'pure_resLSTM_seed_43', \
                 'pure_resLSTM_seed_1024', \
                 'pao_model_seed_7', \
                 'pao_model_seed_43', \
                 'pao_model_seed_1024', \
                 'convnext_seed_7', \
                 'convnext_seed_43', \
                 'convnext_seed_1024', \
                 'encdec_lstm_seed_7', \
                 'encdec_lstm_seed_43', \
                 'encdec_lstm_seed_1024']

wrapped_model_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_good/'
wrapped_models = [wrapped_model_path + 'unet/unet_seed_7/wrapped_model.pt', \
                  wrapped_model_path + 'unet/unet_seed_43/wrapped_model.pt', \
                  wrapped_model_path + 'unet/unet_seed_1024/wrapped_model.pt', \
                  wrapped_model_path + 'squeezeformer/squeezeformer_seed_7/wrapped_model.pt', \
                  wrapped_model_path + 'squeezeformer/squeezeformer_seed_43/wrapped_model.pt', \
                  wrapped_model_path + 'squeezeformer/squeezeformer_seed_1024/wrapped_model.pt', \
                  wrapped_model_path + 'pure_resLSTM/pure_resLSTM_seed_7/wrapped_model.pt', \
                  wrapped_model_path + 'pure_resLSTM/pure_resLSTM_seed_43/wrapped_model.pt', \
                  wrapped_model_path + 'pure_resLSTM/pure_resLSTM_seed_1024/wrapped_model.pt', \
                  wrapped_model_path + 'pao_model/pao_model_seed_7/wrapped_model.pt', \
                  wrapped_model_path + 'pao_model/pao_model_seed_43/wrapped_model.pt', \
                  wrapped_model_path + 'pao_model/pao_model_seed_1024/wrapped_model.pt', \
                  wrapped_model_path + 'convnext/convnext_seed_7/wrapped_model.pt', \
                  wrapped_model_path + 'convnext/convnext_seed_43/wrapped_model.pt', \
                  wrapped_model_path + 'convnext/convnext_seed_1024/wrapped_model.pt', \
                  wrapped_model_path + 'encdec_lstm/encdec_lstm_seed_7/wrapped_model.pt', \
                  wrapped_model_path + 'encdec_lstm/encdec_lstm_seed_43/wrapped_model.pt', \
                  wrapped_model_path + 'encdec_lstm/encdec_lstm_seed_1024/wrapped_model.pt']

for case_prefix, wrapped_model in zip(case_prefixes, wrapped_models):
    run_string = f'''
    python online_run.py \\
            --case_dir {case_dir} \\
            --case_prefix {case_prefix} \\
            --compiled_esm {compiled_esm_path} \\
            --f_torch_model {wrapped_model} \\
            --num_months {num_months} \\
            --job_minutes_per_month {job_minutes_per_month} \\
            --output_frequency {output_frequency} \\
            --email_address {email_address} \\
    '''
    # dedent every line of run_string by one tab
    run_string = textwrap.dedent(run_string)
    print(run_string)
    os.system(run_string)