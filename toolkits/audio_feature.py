import librosa
from python_speech_features import fbank, delta
import  numpy as np
def linear_spec_magnitude_feature(sound_array,n_fft=512,win_length=400,hop_length=160):
    '''

    :param sound_array: raw sound data
    :param n_fft: n_fft
    :param win_length: window length, default: 400 (25ms in sr16k)
    :param hop_length: hop length  default: 160 (10ms in sr16k)
    :return: linear_spec_magnitude
    '''
    s = librosa.stft(sound_array, n_fft=n_fft, win_length=win_length, hop_length=hop_length).T
    mag, _ = librosa.magphase(s)  # magnitude
    return mag.T

def log_fbank_feature(sound_array,n_filter_bank=40,sample_rate=16000):
    filter_banks, energies = fbank(sound_array,
                                   samplerate=sample_rate,
                                   nfilt=n_filter_bank, winlen=0.025)
    filter_banks = 20 * np.log10(np.maximum(filter_banks,1e-5))
    return filter_banks.T






