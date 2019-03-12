import numpy as np
import os

def get_voxceleb2_audio_label_list(data_path, meta_path):
    '''
    :param data_path: voxceleb2  wav dir
    :param meta_path: meta file path
    :return:  audio file list and its label id
    '''
    with open(meta_path) as f:
        lines = f.readlines()
        audiolist = np.array([os.path.join(data_path, s.split()[0]) for s in lines])
        labellist = np.array([int(s.split()[1]) for s in lines])
    return audiolist, labellist