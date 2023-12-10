import torch
import torch.nn as nn

class MatrixFactorizationRMSEModel(nn.Module):
    
    def __init__(self, user_count, item_count, embed_size=40):
        super(MatrixFactorizationRMSEModel, self).__init__()
        
        # MemoryEmbed
        self.user_memory = nn.Embedding(user_count, embed_size)

        # ItemMemory
        self.item_memory = nn.Embedding(item_count, embed_size)

    def _forward(self, userids, itemids):
        # [batch, embedding size]
        user_vec = self.user_memory(userids)

        # [batch, embedding size]
        item_vec = self.item_memory(itemids)

        pred_r = user_vec * item_vec

        return torch.sum(pred_r, dim=1)

    def criterion(self, batch, pred_r):
        """
        Calculate RMSE loss
        """
        ratings = batch[:, 2]
        return torch.sqrt(torch.mean((ratings-pred_r)**2))
    
    def forward(self, batch):
        userids = batch[:, 0]
        itemids = batch[:, 1]

        return self._forward(userids, itemids)
    
    def run_eval(self, batch):
        userids = batch[:, 0]
        itemids = batch[:, 1]

        return self._forward(userids, itemids)
    
class MatrixFactorizationBPRModel(nn.Module):
    
    def __init__(self, user_count, item_count, embed_size=40):
        super(MatrixFactorizationBPRModel, self).__init__()
        
        self.basemodel = MatrixFactorizationRMSEModel(user_count, item_count, embed_size)
        self.sig = nn.Sigmoid()

    def _forward(self, userids, pos_itemids, neg_itemids):
        pos_r = self.basemodel._forward(userids, pos_itemids)
        neg_r = self.basemodel._forward(userids, neg_itemids)

        diff = pos_r - neg_r

        return diff

    def criterion(self, _, vals):
        """
        Calculate BPR loss
        """
        return (1.0 - self.sig(vals)).pow(2).sum()
    
    def forward(self, batch):
        userids = batch[:, 0]
        pos_itemids = batch[:, 1]
        neg_itemids = batch[:, 2]

        return self._forward(userids, pos_itemids, neg_itemids)
    
    def run_eval(self, batch):
        userids = batch[:, 0]
        itemids = batch[:, 1]

        return self.basemodel._forward(userids, itemids)