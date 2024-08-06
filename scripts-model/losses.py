import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, dist, label):

        # print(f"label shape: {label.shape}")
        # print(f"dist shape: {dist.shape}")
        loss_contrastive = torch.mean(1/2*(label) * torch.pow(dist, 2) + 1/2*(1-label) * torch.pow(F.relu(self.margin - dist), 2))

        return loss_contrastive


class TripletLoss(torch.nn.Module):
    """
    Triplet loss function.
    Label is of the form (label_output1, label_output2) where label_output1 says if anchor and output1 are similar, same for label_output2.
    """

    def __init__(self, margin=2.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):

        squarred_distance_1 = (anchor - positive).pow(2).sum(1)
        
        squarred_distance_2 = (anchor - negative).pow(2).sum(1)
        
        triplet_loss = F.relu( self.margin + squarred_distance_1 - squarred_distance_2 ).mean()
        
        return triplet_loss
