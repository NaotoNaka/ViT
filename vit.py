import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

from config import *

class ViT(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()
        self.patchEmbedding = nn.Linear(patchVectorLen,embedVectorLen)
        self.cls = nn.Parameter(torch.zeros(1, 1, embedVectorLen))
        self.positionEmbedding = nn.Parameter(torch.zeros(1, patchTotal + 1, embedVectorLen))
        encoderLayer = TransformerEncoderLayer(
            d_model=embedVectorLen,
            nhead=head,
            dim_feedforward=dim_feedforward,
            activation=activation,
            batch_first=True,
            norm_first=True
        )
        self.transformerEncoder = TransformerEncoder(encoderLayer,layers)
        self.mlpHead=nn.Linear(embedVectorLen,10)

    def patchify(self,img):
        horizontal = torch.stack(torch.chunk(img,splitRow,dim=2),dim=1)
        patches = torch.cat(torch.chunk(horizontal,splitCol,dim=4),dim=1)
        flatPatch = torch.flatten(patches,start_dim=2)
        return flatPatch

    def forward(self,x):
        x=self.patchify(x)
        x=self.patchEmbedding(x)
        clsToken = self.cls.repeat_interleave(x.shape[0],dim=0)
        x=torch.cat((clsToken,x),dim=1)
        x+=self.positionEmbedding
        x=self.transformerEncoder(x)
        x=self.mlpHead(x[:,0,:])
        return x