import os
import librosa
import librosa.display
import toolkits.audio_feature as audio_f
from config.config import default_cfg
from toolkits.voxceleb import get_voxceleb2_audio_label_list
import  matplotlib.pyplot as plt
if __name__=='__main__':
    audio_list,label_list = \
        get_voxceleb2_audio_label_list(data_path=default_cfg['vox2_data_path'],
                                       meta_path=default_cfg['source_root_dir']+'/meta/voxceleb2_val.txt')
    print('audios:',len(audio_list))
    for filepath in audio_list:
        filepath=filepath.replace('.wav','.m4a')
        print(filepath)
        print(os.path.exists(filepath))
        sound, sr= librosa.load(filepath,sr=16000)
        print(sr)
        print('sound len %f s'%(sound.shape[0]/sr))
        mag = audio_f.linear_spec_magnitude_feature(sound)
        logmag = librosa.power_to_db(mag)
        print(logmag.shape)

        melspec = librosa.feature.melspectrogram(sound,
                                                 sr,
                                                 n_fft=512,
                                                 hop_length=512,
                                                 n_mels=128)
        logmelspec = librosa.power_to_db(melspec)
        print(logmelspec.shape)

        plt.figure()
        plt.subplot(3, 1, 1)
        librosa.display.waveplot(sound, sr)
        plt.title('sound')
        plt.subplot(3, 1, 2)
        librosa.display.specshow(logmelspec, sr=sr,x_axis='time', y_axis='mel')
        plt.title('mel')
        plt.subplot(3, 1, 3)
        librosa.display.specshow(logmag, sr=sr, x_axis='time', y_axis='mel')
        plt.title('mag feature')

        plt.show()
        break