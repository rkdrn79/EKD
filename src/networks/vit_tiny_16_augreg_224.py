from torch import nn
import torch.nn.functional as F
from .vit_original import VisionTransformer, _load_weights


# It can handle bsize of 50 to 60

class Vit_tiny_16_augreg_224(nn.Module):

    def __init__(self, num_classes=100, pretrained=False):
        super().__init__()

        #import ipdb; ipdb.set_trace()
        self.vit = VisionTransformer(embed_dim=192, num_heads=3, num_classes=0)

        # last classifier layer (head) with as many outputs as classes
        self.fc = nn.Linear(in_features=192, out_features=num_classes, bias=True)
        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = 'fc'

    def forward(self, x):
        h = self.fc(self.vit(x))
        return h

class Vit_tiny_4_augreg_32(nn.Module):
    def __init__(self, num_classes=100, pretrained=False):
        super().__init__()

        self.vit = VisionTransformer(
            img_size=32,
            patch_size=4,
            in_chans=3,
            num_classes=0,
            embed_dim=192,
            depth=12,
            num_heads=3,
            mlp_ratio=4,
            weight_init=''
        )

        self.fc = nn.Linear(in_features=192, out_features=num_classes, bias=True)
        self.head_var = 'fc'

    def forward(self, x):
        x = self.vit(x)
        x = self.fc(x)
        return x


def vit_tiny_16_augreg_224(num_out=100, pretrained=False):
    if pretrained:
        return Vit_tiny_16_augreg_224(num_out, pretrained)
    else:
        raise NotImplementedError
    assert 1==0, "you should not be here :/"

def vit_tiny_4_augreg_32(num_out=100, pretrained=False):
    if pretrained:
        return Vit_tiny_4_augreg_32(num_out, pretrained)
    else:
        raise NotImplementedError
    assert 1==0, "you should not be here :/"

