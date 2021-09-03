import pandas as pd
from mne import create_info
from mne.io import RawArray


def load_raw(filename, sfreq=256., ch_ind=[0, 1, 2, 3],
             stim_ind=5, replace_ch_names=None):
    '''
        FUNCIÓN para creación del objeto Raw a partir del fichero con los datos del experimento.
    '''
    n_channel = len(ch_ind)
    data = pd.read_csv(filename)

    if "Timestamp" in data.columns:
        del data['Timestamp']
    if "Time" in data.columns:
        del data['Time']
    if stim_ind is not None:
        ch_names = list(data.columns)[0:n_channel] + ['Stim']
    else:
        ch_names = list(data.columns)[0:n_channel]

    if replace_ch_names is not None:
        ch_names = [c if c not in replace_ch_names.keys()
                    else replace_ch_names[c] for c in ch_names]

    montage = 'standard_1020'
    if stim_ind is not None:
        ch_types = ['eeg'] * n_channel + ['stim']
        data = data.values[:, ch_ind + [stim_ind]].T
    else:
        ch_types = ['eeg'] * n_channel
        data = data.values[:, ch_ind].T
        
    data[:-1] *= 1e-6

    info = create_info(ch_names=ch_names, ch_types=ch_types,
                       sfreq=sfreq)
    raw = RawArray(data=data, info=info)
    print(len(data))
    print(len(info["ch_names"]))
    raw.set_montage(montage)

    return raw