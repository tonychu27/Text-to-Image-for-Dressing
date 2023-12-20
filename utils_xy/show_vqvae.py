import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

def show(img):
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.axis('off')
    plt.pause(1.5)
    plt.close()
    

def pipeline(x_origin: torch.Tensor, x_rec: torch.Tensor):
    x_origin_copy, x_rec_copy = x_origin.detach().clone(), x_rec.detach().clone()
    
    img_cat = torch.cat([x_origin_copy, x_rec_copy], dim=3)
    img_cat = ((img_cat + 1) / 2)
    img_cat = img_cat.clamp_(0, 1)
    show(make_grid(img_cat, nrow=1, padding= 4))