import torch
import torch.nn as nn

class MatrixFactorizationModel(nn.Module):
    
    def __init__(self, user_count, item_count, embed_size=800):
        """
        Constructs the user/item memories and user/item external memory/outputs

        Also add the embedding lookups
        """
        super(MatrixFactorizationModel, self).__init__()
        
        # MemoryEmbed
        self.user_memory = nn.Embedding(user_count, embed_size)

        # ItemMemory
        self.item_memory = nn.Embedding(item_count, embed_size)

    def forward(self, userids, itemids):
        """
        Construct the model; main part of it goes here
        """
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