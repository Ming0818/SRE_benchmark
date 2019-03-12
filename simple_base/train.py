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


def eval(model, dataloader):
    rightN=0
    for data,label in dataloader:
        data = data.cuda()
        pred = model(data)
        pred = np.argmax(pred.data.cpu().numpy(),1)
        gt = label.numpy()
        rightN+=(pred==gt).sum()
    acc = rightN/len(dataloader.dataset)
    print('acc:',acc)
    return acc
def adjust_learning_rate(base_lr, optimizer, epoch):
    base =6
    # if epoch>base:
    #     base=8
    lr = base_lr * (0.2 ** (epoch // base))
    print('lr:',lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_v1(cfg ):
    model = resnet18_slim_model().cuda()
    trn_ds = Vox2Dataset(data_path=cfg['vox2_data_path'],
                         meta_path=cfg['source_root_dir'] +
                                   '/meta/voxceleb2_val.txt',
                         mode='test')
    val_ds = Vox2Dataset(data_path=cfg['vox2_data_path'],
                         meta_path=cfg['source_root_dir'] +
                                   '/meta/voxceleb2_val.txt',
                         mode='test')
    wokers = 4
    trn_dl = DataLoader(trn_ds,shuffle=True, batch_size=256, num_workers=wokers)
    val_dl = DataLoader(val_ds,shuffle=False, batch_size=256, num_workers=wokers)
    base_lr=1e-3
    optim = torch.optim.Adam(model.parameters(),base_lr)##
    #optim = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9)
    best_eval_score_avg = 0.0
    print('Begin training......')
    early_stop_counter = 0
    num_epochs=20
    for epoch in range(num_epochs):
        total_loss = 0
        rightN=0
        for data,label in tqdm.tqdm(trn_dl):
            optim.zero_grad()
            data  = data.cuda()
            pred = model(data)
            label = label.cuda()

            loss = F.cross_entropy(pred,   label,reduction='none')
            loss,_ = loss.topk(k=int(loss.size(0)*0.9))
            loss = loss.mean()
            pred = np.argmax(pred.data.cpu().numpy(), 1)
            gt =   label.cpu().numpy()
            rightN += (pred == gt).sum()
            loss.backward()
            optim.step()
            total_loss += loss.item()* label.size(0)

        NN=len(trn_dl.dataset)
        total_loss/=NN
        print('epoch %d, \ttrain_loss: %.3f,'
                     ' train_score: %.3f' % (epoch, total_loss,rightN/NN))
        adjust_learning_rate(base_lr, optim, epoch)
        if epoch >= 0:
            model.train(False)
            eval_score   = eval(model, val_dl)
            model.train(True)
            if eval_score >= best_eval_score_avg:
                early_stop_counter = 0
                # model_path = os.path.join(output_dir, 'model_best.pth')
                # torch.save(model.state_dict(), model_path)
                best_eval_score_avg = eval_score
            else:
                early_stop_counter += 1
            print('epoch %d,' % (epoch) +
                         '\teval score: %.2f ' % (100 * eval_score) +
                         '( best:  %.2f)' % (100 * best_eval_score_avg))
            # if early_stop_counter>early_stop_n:
            #     break
    print('**************************************************')

def train(cfg):
    model = resnet18_slim_model().cuda()
    metric = Top_N_Acc(topn=1)
    trn_ds = Vox2Dataset(data_path=cfg['vox2_data_path'],
                         meta_path=cfg['source_root_dir']+
                                   '/meta/voxceleb2_train.txt',
                         mode='train')
    val_ds = Vox2Dataset(data_path=cfg['vox2_data_path'],
                           meta_path=cfg['source_root_dir'] +
                                     '/meta/voxceleb2_val.txt',
                         mode='test')
    wokers=4
    trn_dl = DataLoader(trn_ds,batch_size=512,shuffle=True, num_workers=wokers, pin_memory=True)
    val_dl = DataLoader(val_ds,batch_size=512,shuffle=False, num_workers=wokers, pin_memory=True)

    loss_fn = F.cross_entropy

    adam = partial(optim.Adam, lr=1e-2)
    writer = SummaryWriter(cfg['log_dir']+'/log_simple_base')
    learner = Learner(model=model, trn_dl=trn_dl, val_dl=val_dl,
                      optim_fn=adam,
                      metrics=[metric], loss_fn=loss_fn,
                      callbacks=[], writer=writer)
    #to_fp16(learner, 512)
    learner.callbacks.append(SaveBestModel(learner, small_better=False, name='simple_base',
                                           model_dir=cfg['log_dir']+'/log_simple_base'))

    epoches = 40
    max_lr = 1e-3
    lrs = np.linspace(max_lr, max_lr / 100, num=epoches * len(trn_dl))
    lr_sched = LrScheduler(learner.optimizer, lrs)
    learner.fit(epoches,lr_sched)

if __name__=='__main__':
    train(default_cfg)



