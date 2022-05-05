from slow import *
from slowfast import *
from x3d import *
from mvit import *
from vimpac import *
from mae import *


__models__ = {'slow': SlowR50,
              'slowfast': SlowFastR50,
              'x3d': X3D,
              'mvit': MViT,
              'vimpac': VIMPAC,
              'mae': VideoMAE}


def get_model(model):
    return __models__[model]


if __name__ == '__main__':
    # Set random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)

    frame = torch.ones(2, 5, 3, 128, 128)
    model = VIMPAC(num_classes=4, pretrained=True, freeze=True)
    output = model(frame)
    print(output)
