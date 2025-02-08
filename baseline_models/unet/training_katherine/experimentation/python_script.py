import sys
import os
import re
import glob
import numpy as np
from data_stream import get_herbie_data
from dask.distributed import Client

year = int(sys.argv[1])
month = int(sys.argv[2])
fxx = int(sys.argv[3])
var = sys.argv[4]

herbie_collector = get_herbie_data()
herbie_collector.fxx = int(fxx)
hrrr_path = '/global/cfs/cdirs/m4334/jerry/wind_forecasting/HRRR_{}'.format(str(fxx).zfill(2))
herbie_collector.path_out = hrrr_path
monthdays = get_herbie_data.get_days_of_month(year, month)
varpath = os.path.join(herbie_collector.path_out, var, 'hrrr')
daypaths = [os.path.join(varpath, day) for day in monthdays]

def create_day_arr(daypath):
    print(f'loading {daypath}')
    if not os.path.exists(daypath):
        raise FileNotFoundError(f'{daypath} is missing')
    arrs = []
    day_files = herbie_collector.sort_day_files(daypath)
    if fxx == 0 and len(day_files) != 24:
        raise ValueError(f'Expected 24 files, but found {len(day_files)} in {daypath}')
    elif fxx == 18 and len(day_files) != 24:
        raise ValueError(f'Expected 24 files, but found {len(day_files)} in {daypath}')
    elif fxx == 24 and len(day_files) != 4:
        raise ValueError(f'Expected 4 files, but found {len(day_files)} in {daypath}')
    for file_idx in range(len(day_files)):
        try:
            message = herbie_collector.read_message(day_files[file_idx])
        except Exception as e:
            print(f'Error reading {day_files[file_idx]}: {e}')
            try:
                print('attempting redownload')
                match = re.search(r'/(\d{4})(\d{2})(\d{2})/.*\.t(\d{2})z\.', day_files[file_idx])
                if match:
                    year, month, day, hour = match.groups()
                    timestamp = f'{year}-{month}-{day} {hour}:00'
                    print(timestamp)
                else:
                    print("No match found")
                herbie_collector.download_sample(timestamp, var)
                print('redownload complete')
                print('rereading file')
                message = herbie_collector.read_message(day_files[file_idx])
                print('reread complete')
            except:
                print('redownload failed')
                raise
        arrs.append(message.values)
    day_arr = np.stack(arrs, axis = 0)
    print(f'loaded {daypath}')
    return day_arr

if __name__ == '__main__':
    print('creating client')
    client = Client()
    futures = client.map(create_day_arr, daypaths)

    
    arr_list = client.gather(futures)
    client.close()
    print('closed client')
    arrays_dict = {name[-8:]: array for name, array in zip(daypaths, arr_list)}
    np.savez(os.path.join(herbie_collector.path_out, var, f'{var}_{str(year)}_{str(month).zfill(2)}_arr.npz'), **arrays_dict)