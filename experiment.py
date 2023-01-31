import torch
import torch.nn as nn
import torch.nn.functional as F


class Experiment:

    def __init__(self, model, train_loader, test_loader, optimizer, criterion=F.cross_entropy, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def training_step(self, batch):
        (ll, lh, hl, hh), labels = batch
        ll, lh, hl, hh, labels = ll.to(self.device), lh.to(self.device), hl.to(
            self.device), hh.to(self.device), labels.to(self.device)
        out = self.model((ll, lh, hl, hh))
        loss = self.criterion(
            out, labels, label_smoothing=0.1)

        return loss

    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            for batch in self.train_loader:
                loss = self.training_step(batch)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    def validation_step(self, batch):
        (ll, lh, hl, hh), labels = batch
        ll, lh, hl, hh, labels = ll.to(self.device), lh.to(self.device), hl.to(
            self.device), hh.to(self.device), labels.to(self.device)
        out = self.model((ll, lh, hl, hh))
        loss = self.criterion(out, labels)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        return {'val_loss': epoch_loss.item()}

    def validate(self):
        self.model.eval()
        outputs = [self.validation_step(batch) for batch in self.test_loader]
        return self.validation_epoch_end(outputs)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def accuracy(self, dataloader):
        count = 0
        self.model.eval()
        outs = []
        labelss = []
        for batch in dataloader:
            (ll, lh, hl, hh), labels = batch
            ll, lh, hl, hh, labels = ll.to(self.device), lh.to(self.device), hl.to(
                self.device), hh.to(self.device), labels.to(self.device)
            out = self.model((ll, lh, hl, hh))
            out = torch.max(out, dim=1).indices
            outs += out.tolist()
            labelss += labels.tolist()
    
        import sklearn.metrics as metrics
        print(metrics.confusion_matrix(labelss, outs))



# 