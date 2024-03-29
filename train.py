import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from experiment import Experiment
from get_dataset import MoireDataset
from modules.mCNN import mCNN
import os 
# torch.cuda.set_per_process_memory_fraction(0.98, 0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if __name__ == '__main__':
    train_ds, test_ds = random_split(MoireDataset, [len(MoireDataset) - 200, 200])
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)
    torch.save(test_ds, 'test_ds.pth')
    torch.save(train_ds, 'train_ds.pth')
    model = mCNN(2).to(device)
    optimizer = Adam(model.parameters(), lr=0.00003)
    experiment = Experiment(model, train_loader, test_loader, optimizer, device=device)
    if os.path.exists(os.path.join('model', 'model.pth')):
        experiment.load(os.path.join('model', 'model.pth'))
    experiment.train(10)

    experiment.save(os.path.join('model', 'model.pth'))
    
    experiment.accuracy(test_loader)
    # print(experiment.validate())