import torch
from  torch.utils.data  import DataLoader,Dataset
from  toolkits.voxceleb import get_voxceleb2_audio_label_list
from toolkits.audio_feature import linear_spec_magnitude_feature,log_fbank_feature
import numpy as np
import librosa
class Vox2Dataset(Dataset):
    def __init__(self,
                 data_path,
                 meta_path,
                 is_just_calc_fea=False,
                 mode='train',
                 train_spec_len=256,## 3s
                 feature_ext_fn=log_fbank_feature):
        super(Vox2Dataset,self).__init__()
        self.audios,self.labels=\
            get_voxceleb2_audio_label_list(data_path=data_path,meta_path=meta_path)
        self.audios = [f.replace('.wav','.m4a') for f in self.audios]
        self.feature_fn=feature_ext_fn
        self.mode=mode
        self.train_spec_len=train_spec_len
        self.is_just_calc_fea =is_just_calc_fea
    def __len__(self):
        return  len(self.audios)

    def _norm(self,fea):
        mu = np.mean(fea, 0, keepdims=True)
        std = np.std(fea, 0, keepdims=True)
        return (fea - mu) / (std + 1e-12)

    def __getitem__(self, inx):
        audio_file = self.audios[inx]
        label = self.labels[inx]
        if self.is_just_calc_fea:
            sound, sr = librosa.load(audio_file, sr=16000)
            # we only keep max 8s speech
            if sound.shape[0] > 8 * 16000:
                sound = sound[:8 * 16000]

            mag_fea = self.feature_fn(sound)
            freq, time = mag_fea.shape

            npy_file = audio_file.replace('.m4a', '.npy')
            np.save(npy_file, mag_fea)
            return 'ha'

        npy_file = audio_file.replace('.m4a', '.npy')
        mag_fea = np.load(npy_file)

        freq, time = mag_fea.shape
        if time > self.train_spec_len:
            if self.mode == 'train':
                randtime = np.random.randint(0, time - self.train_spec_len)
                mag_fea = mag_fea[:, randtime:randtime + self.train_spec_len]
            else:
                mag_fea = mag_fea[:, 0:self.train_spec_len]
        else:
            padd_fea = np.zeros((freq, self.train_spec_len), np.float32)
            padd_fea[:, :time] = mag_fea
            mag_fea = padd_fea

        #print(mag_fea.shape)

        mag_fea_norm = self._norm(mag_fea)
        return torch.from_numpy(mag_fea_norm).unsqueeze(0).float(),\
               torch.LongTensor([label]).squeeze()

from config.config import default_cfg as cfg
import tqdm
if __name__=='__main__':
    ds = Vox2Dataset(data_path=cfg['vox2_data_path'],
                         meta_path=cfg['source_root_dir'] +
                                   '/meta/voxceleb2_val.txt')
    wokers = 6
    dl = DataLoader(ds, batch_size=1, num_workers=wokers, pin_memory=True)
    for _ in tqdm.tqdm(dl):
        pass