from simple_base.model import resnet18_slim_model
from config.config import default_cfg
from simple_base.data import Vox2Dataset
from toolkits.metrics import Top_N_Acc
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from  lightai.core import *
from  lightai.train import *
import tqdm
import librosa
import toolkits
from toolkits.audio_feature import log_fbank_feature
def top_1_acc(model, dataloader):
    rightN=0
    for data,label in tqdm.tqdm(dataloader):
        data = data.cuda()
        pred = model(data)
        pred = np.argmax(pred.data.cpu().numpy(),1)
        gt = label.numpy()
        rightN+=(pred==gt).sum()
    acc = rightN/len(dataloader.dataset)
    print('top-1 acc:',acc)
    return acc

def validate(cfg):
    model = resnet18_slim_model().cuda()
    model.load_state_dict(torch.load(cfg['log_dir']+'/log_simple_base/simple_base')['model'])
    model.eval()
    val_ds = Vox2Dataset(data_path=cfg['vox2_data_path'],
                           meta_path=cfg['source_root_dir'] +
                                     '/meta/voxceleb2_val.txt',
                         mode='test')
    wokers=4
    val_dl = DataLoader(val_ds,batch_size=512,shuffle=False, num_workers=wokers, pin_memory=True)

    top_1_acc(model,val_dl)


def verify_vox1(cfg,test_type='normal', use_fea_len=512):
    if test_type == 'normal':
        verify_list = np.loadtxt( cfg['source_root_dir']
                                  +'/meta/voxceleb1_veri_test_fixed.txt', str)
    elif test_type == 'hard':
        verify_list = np.loadtxt(cfg['source_root_dir']
                                  +'/meta/voxceleb1_veri_test_hard_fixed.txt', str)
    elif test_type == 'extend':
        verify_list = np.loadtxt(cfg['source_root_dir']
                                  +'/meta/voxceleb1_veri_test_extended_fixed.txt', str)
    else:
        raise IOError('==> unknown test type.')

    verify_label = np.array([int(i[0]) for i in verify_list])
    list1 = np.array([os.path.join(cfg['vox1_data_path'], i[1]) for i in verify_list])
    list2 = np.array([os.path.join(cfg['vox1_data_path'], i[2]) for i in verify_list])
    total_list = np.concatenate((list1, list2))
    unique_list = np.unique(total_list)

    model = resnet18_slim_model(is_ext_fea=True).cuda()
    model.load_state_dict(torch.load(cfg['log_dir'] + '/log_simple_base/simple_base')['model'])
    model.eval()

    feas, scores, labels = [], [], []
    for filename in tqdm.tqdm(unique_list):
        sound, sr = librosa.load(filename, sr=16000)
        # we only keep max 8s speech
        if sound.shape[0] > 8 * 16000:
            sound = sound[:8 * 16000]
        fbank = log_fbank_feature(sound)
        freq, time = fbank.shape
        if time >use_fea_len:
            fbank = fbank[:, 0:use_fea_len]
        else:
            padd_fea = np.zeros((freq, use_fea_len), np.float32)
            padd_fea[:, :time] = fbank
            fbank = padd_fea
        mu = np.mean(fbank, 0, keepdims=True)
        std = np.std(fbank, 0, keepdims=True)
        fbank= (fbank - mu) / (std + 1e-12)
        input = np.expand_dims(np.expand_dims(fbank, 0), 0)
        input = torch.from_numpy(input).float().cuda()
        with torch.no_grad():
            out = model(input).cpu().numpy()
        feas+=[out]



    for c, (p1, p2) in enumerate(zip(list1, list2)):
        ind1 = np.where(unique_list == p1)[0][0]
        ind2 = np.where(unique_list == p2)[0][0]
        v1 = feas[ind1]
        v2 = feas[ind2]
        scores += [np.sum(v1 * v2)/ (np.sqrt(np.sum(v1*v1)) *np.sqrt(np.sum(v2*v2)))]
        labels += [verify_label[c]]
        #print('scores : {}, gt : {}'.format(scores[-1], verify_label[c]))

    scores = np.array(scores)
    labels = np.array(labels)
    eer, thresh = toolkits.metrics.calculate_eer(labels, scores)
    print('==>  EER: {}--Thresh: {}'.format(eer,thresh))








if __name__=='__main__':
    print('validate on vox2 val...')
    validate(default_cfg)

    print('verify on vox1...')
    verify_vox1(default_cfg)


