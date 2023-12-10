import torch
import torch.nn as nn

class MatrixFactorizationModel(nn.Module):
    
    def __init__(self, user_count, item_count, embed_size=40):
        super(MatrixFactorizationModel, self).__init__()
        
        # MemoryEmbed
        self.user_memory = nn.Embedding(user_count, embed_size)

        # ItemMemory
        self.item_memory = nn.Embedding(item_count, embed_size)

    def forward(self, userids, itemids):
        # [batch, embedding size]
        user_vec = self.user_memory(userids)

        # [batch, embedding size]
        item_vec = self.item_memory(itemids)

        pred_r = user_vec * item_vec

        return torch.sum(pred_r, dim=1)

    def criterion(self, r, pred_r):
        """
        Calculate RMSE loss
        """
        return torch.sqrt(torch.mean((r-pred_r)**2))
    
class MatrixFactorizationBPRModel(nn.Module):
    
    def __init__(self, user_count, item_count, embed_size=40):
        super(MatrixFactorizationBPRModel, self).__init__()
        
        self.mfmodel = MatrixFactorizationModel(user_count, item_count, embed_size)
        self.sig = nn.Sigmoid()

    def forward(self, userids, pos_itemids, neg_itemids):
        pos_r = self.mfmodel(userids, pos_itemids)
        neg_r = self.mfmodel(userids, neg_itemids)

        diff = pos_r - neg_r

        return diff

    def criterion(self, vals):
        """
        Calculate BPR loss
        """
        return (1.0 - self.sig(vals)).pow(2).sum()