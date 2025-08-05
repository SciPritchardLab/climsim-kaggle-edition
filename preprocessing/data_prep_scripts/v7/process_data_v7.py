import argparse
import xarray as xr
from climsim_utils.data_utils import *

def main(regexps, data_split, stride_sample, start_idx, save_h5, save_path):
    grid_path = '../../../grid_info/ClimSim_low-res_grid-info.nc'
    norm_path = '../../normalizations/'

    grid_info = xr.open_dataset(grid_path)
    input_mean = xr.open_dataset(norm_path + 'inputs/input_mean_v7_pervar.nc')
    input_max = xr.open_dataset(norm_path + 'inputs/input_max_v7_pervar.nc')
    input_min = xr.open_dataset(norm_path + 'inputs/input_min_v7_pervar.nc')
    output_scale = xr.open_dataset(norm_path + 'outputs/output_scale_std_lowerthred_v7.nc')

    data = data_utils(grid_info = grid_info, 
                    input_mean = input_mean, 
                    input_max = input_max, 
                    input_min = input_min, 
                    output_scale = output_scale,
                    qinput_log = False,
                    normalize = False,
                    input_abbrev='ml2steploc',
                    output_abbrev='mlo',
                    save_h5 = True)
    print(data_split)
    if data_split == 'test':
        data.data_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/_test/'
    else:
        data.data_path = '/pscratch/sd/j/jerrylin/hugging/E3SM-MMF_ne4/train/'
    data.set_to_v7_vars()

    # if "," in regexps, then split it to a list of strings
    # regexps must be str,
    
    # if "," in regexps:
    #     regexps = regexps.split(',')
    # else:
    #     regexps = [regexps]
    print(regexps)
    data.set_regexps(data_split = data_split, regexps = regexps)
    data.set_stride_sample(data_split = data_split, stride_sample = stride_sample)
    data.set_filelist(data_split = data_split, start_idx = start_idx)
    #if savepath not exist, create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data.save_as_npy(data_split = data_split, save_path = save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process E3SM-MMF data.')
    parser.add_argument('regexps', type=str, nargs='+', help='Regular expressions for selecting data files.')
    parser.add_argument('--data_split', type=str, required=True, help='Data split (train, val, scoring or test)')
    parser.add_argument('--stride_sample', type=int, required=True, help='Stride sample.')
    parser.add_argument('--start_idx', type=int, required=True, help='Start index of the data file to be processed.')
    parser.add_argument('--save_h5', type=bool, required=True, help='Save as h5 file.')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save processed data.')
    args = parser.parse_args()
    main(args.regexps, args.data_split, args.stride_sample, args.start_idx, args.save_h5, args.save_path)