import os
import sys
import subprocess
import textwrap

# copy online_ensemble_template.sh to online_ensemble.sh
model_ensemble = 'short_runs'
ensemble_dir = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/online_runs/climsim3_ensembles_v6'
case_dir = f'{ensemble_dir}/{model_ensemble}'
compiled_esm_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/online_runs/climsim3_ensembles_v6/empty_case/build/e3sm.exe'
long_run = False
email_address = 'jerryL9@uci.edu'

case_prefixes = ['unet_seed_43', 'squeezeformer_seed_43', 'pure_resLSTM_seed_43', 'pao_model_seed_43', 'convnext_seed_43', 'encdec_lstm_seed_43']
case_prefixes = ['pao_model_seed_43']

wrapped_model_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/saved_models/climsim3_ensembles_v6/'
wrapped_models = [wrapped_model_path + 'unet/unet_seed_43/wrapped_model.pt', \
                  wrapped_model_path + 'squeezeformer/squeezeformer_seed_43/wrapped_model.pt', \
                  wrapped_model_path + 'pure_resLSTM/pure_resLSTM_seed_43/wrapped_model.pt', \
                  wrapped_model_path + 'pao_model/pao_model_seed_43/wrapped_model.pt', \
                  wrapped_model_path + 'convnext/convnext_seed_43/wrapped_model.pt', \
                  wrapped_model_path + 'encdec_lstm/encdec_lstm_seed_43/wrapped_model.pt']
wrapped_models = [wrapped_model_path + 'pao_model/pao_model_seed_43/wrapped_model.pt']

for case_prefix, wrapped_model in zip(case_prefixes, wrapped_models):
    if long_run:
        run_string = f'''
        python online_run.py \\
                --case_dir '{case_dir}' \\
                --case_prefix '{case_prefix}' \\
                --compiled_esm '{compiled_esm_path}' \\
                --f_torch_model '{wrapped_model}' \\
                --email_address '{email_address}' \\
                --long_run
        '''
        run_string = textwrap.dedent(run_string)
    else:
        run_string = f'''
        python online_run.py \\
                --case_dir '{case_dir}' \\
                --case_prefix '{case_prefix}' \\
                --compiled_esm '{compiled_esm_path}' \\
                --f_torch_model '{wrapped_model}' \\
                --email_address '{email_address}' \\
        '''
        run_string = textwrap.dedent(run_string)
    # dedent every line of run_string by one tab
    run_string = textwrap.dedent(run_string)
    print(run_string)
    os.system(run_string)