import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import matplotlib.pyplot as plt
import os
from experiment import Experiment
from get_dataset import MoireDataset
from modules.mCNN import mCNN
import pywt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if __name__ == '__main__':
    test_ds = MoireDataset
    test_ds = torch.load('./test_ds.pth')
    count = 0
    length = len(test_ds)
    outs = []
    labelss = []
    scores = []
    for i in range(0, length, 200):
        test_set = Subset(test_ds, range(i, min(i + 200, length)))
        test_loader = DataLoader(test_set, batch_size=3, shuffle=False)

        model = mCNN(2).to(device)
        model.load_state_dict(torch.load('./model/model.pth'))
        model.eval()

        label_names = ['Moire', 'No Moire']

        experiment = Experiment(model, None, test_loader, None, device=device)


        for batch in test_loader:
            (ll, lh, hl, hh), labels = batch
            ll, lh, hl, hh, labels = ll.to(device), lh.to(device), hl.to(
                device), hh.to(device), labels.to(device)
            out = model((ll, lh, hl, hh))
            a = torch.max(out, dim=1).indices.tolist()
            outs += a
            scores += out[:,1].tolist()
            b = labels.tolist()
            labelss += b
            for j in range(len(labels)):
                image = pywt.idwt2((ll[j].cpu().numpy(), (lh[j].cpu().numpy(
                ), hl[j].cpu().numpy(), hh[j].cpu().numpy())), 'haar')
                image = image.reshape(1024, -1)
                if a[j] != b[j]:
                    
                    save_path = os.path.join(".", 'wrong_detection', 'true_' + \
                        label_names[a[j]] + '_' + 'pred_' + \
                        label_names[b[j]] + '_' + str(count) + '.png')
                    count += 1
                    plt.imsave(save_path, image, cmap='gray')
            
        # confusion_mat = metrics.confusion_matrix(a, b, labels=[0, 1])
        # try:
        #     old_confusion_mat = np.load('confusion_mat.npy')
        #     confusion_mat = confusion_mat + old_confusion_mat
        # except:
        #     pass
        # np.save('confusion_mat.npy', confusion_mat)
        # print("Confusion Matrix:", confusion_mat)
    print("Confusion Matrix:", metrics.confusion_matrix(outs, labelss))
    print("Accuracy:", metrics.accuracy_score(outs, labelss))
    print("Precision:", metrics.precision_score(outs, labelss))
    print("Recall:", metrics.recall_score(outs, labelss))
    print("F1 Score:", metrics.f1_score(outs, labelss))
    print("AUC:", metrics.roc_auc_score(outs, labelss))
    
    # plot auc roc curve
    fpr, tpr, thresholds = metrics.roc_curve(labelss, scores, pos_label=1)
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig('auc_roc_curve_data_v2.png')
    plt.show()

    # plot precision recall curve
    precision, recall, thresholds = metrics.precision_recall_curve(
        labelss, scores, pos_label=1)
    plt.plot(recall, precision, color='orange', label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()

    plt.savefig('precision_recall_curve_v2.png')
    plt.show()

    scores = np.array(scores)
    labelss = np.array(labelss)
    np.save('scores_data_v2.npy', scores)
    np.save('labels_data_v2.npy', labelss)