# from tqdm.auto import tqdm
from tqdm import tqdm
import numpy as np
import torch

class Trainer:
    def __init__(self, dataset, model, optimizer, metrics, epochs=10, batch_size=32, device='cpu'):
        self.dataset = dataset
        self.model = model.to(device)
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.metrics = metrics
        
        self.train_log = []
        self.test_log = []

    def train(self, evaluate=False):
        self.model.train()
        self.model.zero_grad()

        for epoch in range(self.epochs):
            epoch_losses = []
            pbar = tqdm(
                self.dataset.train_generator(self.batch_size), 
                dynamic_ncols=True, 
                total=(
                    self.dataset.train_size//self.batch_size + 
                    (1 if self.dataset.train_size % self.batch_size > 0 else 0)
                ),
                leave=False
            )
            
            for batch in pbar:
                self.optimizer.zero_grad()

                batch = torch.LongTensor(batch).to(self.device).to(self.device)

                userids = batch[:, 0]
                itemids = batch[:, 1]
                ratings = batch[:, 2]

                preds = self.model(userids, itemids)
                loss = self.model.criterion(ratings, preds)

                loss.backward()

                self.optimizer.step()

                epoch_losses.append(loss.item())
                pbar.set_description(u"[{}] Loss: {:,.4f} » ".format(epoch, loss.item()))
            
            pbar.reset()
            mean_epoch_loss = np.mean(epoch_losses)
            print("Epoch {}: Avg Loss/Batch {:<20,.6f}".format(epoch, mean_epoch_loss))
            
            self.train_log.append(mean_epoch_loss)

            if evaluate:
                self.test()

    def test(self):
        self.model.zero_grad()
        self.model.eval()

        preds = []
        with torch.no_grad():
            pbar = tqdm(self.dataset.test_generator(), total=self.dataset.test_size, leave=False)

            for uid, pos_iid, neg_iids in pbar:
                batch = list(zip([uid]*101, neg_iids+[pos_iid], [0]*100+[1]))
                batch = torch.LongTensor(batch).to(self.device)

                userids = batch[:, 0]
                itemids = batch[:, 1]
                ratings = batch[:, 2]

                preds.append(self.model(userids, itemids).cpu().numpy())

            pbar.reset()

        compiled = {}
        for m in self.metrics:
            func = self.metrics[m][0]
            args = self.metrics[m][1]
            result = func(preds, **args)
            compiled[m] = result
            print(f"{m}: {result}")

        self.test_log.append(compiled)

class BPRTrainer(Trainer):
    def train(self, evaluate=False):
        self.model.train()
        self.model.zero_grad()

        for epoch in range(self.epochs):
            epoch_losses = []
            pbar = tqdm(
                self.dataset.train_generator(self.batch_size), 
                dynamic_ncols=True, 
                total=(
                    self.dataset.train_size//self.batch_size + 
                    (1 if self.dataset.train_size % self.batch_size > 0 else 0)
                ),
                leave=False
            )
            
            for batch in pbar:
                self.optimizer.zero_grad()

                batch = torch.LongTensor(batch).to(self.device).to(self.device)

                userids = batch[:, 0]
                pos_itemids = batch[:, 1]
                neg_itemids = batch[:, 2]

                vals = self.model(userids, pos_itemids, neg_itemids)
                loss = self.model.criterion(vals)

                loss.backward()

                self.optimizer.step()

                epoch_losses.append(loss.item())
                pbar.set_description(u"[{}] Loss: {:,.4f}".format(epoch, loss.item()))
            
            pbar.reset()
            mean_epoch_loss = np.mean(epoch_losses)
            print("Epoch {}: Avg Loss/Batch {:<20,.6f}".format(epoch, mean_epoch_loss))
            
            self.train_log.append(mean_epoch_loss)

            if evaluate:
                self.test()

    def test(self):
        self.model.zero_grad()
        self.model.eval()

        preds = []
        with torch.no_grad():
            pbar = tqdm(self.dataset.test_generator(), total=self.dataset.test_size, leave=False)

            for uid, pos_iid, neg_iids in pbar:
                batch = list(zip([uid]*101, neg_iids+[pos_iid], [0]*100+[1]))
                batch = torch.LongTensor(batch).to(self.device)

                userids = batch[:, 0]
                itemids = batch[:, 1]
                ratings = batch[:, 2]

                preds.append(self.model.mfmodel(userids, itemids).cpu().numpy())

            pbar.reset()

        compiled = {}
        for m in self.metrics:
            func = self.metrics[m][0]
            args = self.metrics[m][1]
            result = func(preds, **args)
            compiled[m] = result
            print(f"{m}: {result}")

        self.test_log.append(compiled)
