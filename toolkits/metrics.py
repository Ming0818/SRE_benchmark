from sklearn.metrics import accuracy_score
import numpy as np

class Top_N_Acc():
    def __init__(self, topn=1):
        self.predicts = []
        self.targets = []
        self.topn = topn

    def __call__(self, predict, target):
        """
        predict and target are in batch
        """
        predict = predict.tolist()
        target = target.tolist()
        self.predicts.extend(predict)
        self.targets.extend(target)

    def res(self):
        pred = np.argmax(np.array(self.predicts),1)
        print(pred.shape)
        acc= accuracy_score(self.targets,pred)
        self.predicts.clear()
        self.targets.clear()
        return  acc
        # print(np.array(self.predicts).shape)
        # best_n = np.argsort(np.array(self.predicts), axis=1)[:, -self.topn:]
        # # ts = np.argmax(self.targets, axis=1)
        # ts = np.array(self.targets)
        # successes = 0
        # for i in range(ts.shape[0]):
        #     if ts[i] in best_n[i, :]:
        #         successes += 1
        # self.predicts.clear()
        # self.targets.clear()
        # return float(successes) / ts.shape[0]

def calculate_eer(y, y_score):
    # y denotes groundtruth scores,
    # y_score denotes the prediction scores.
    from scipy.optimize import brentq
    from sklearn.metrics import roc_curve
    from scipy.interpolate import interp1d
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh
