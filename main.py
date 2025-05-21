import nst

import matplotlib.pyplot as plt
import numpy as np
import torch

def imshow(imgs):
    imgs = imgs / 2 + 0.5
    npimgs = imgs.numpy()
    plt.imshow(np.transpose(npimgs,(1,2,0)))
    plt.show()

content = "" # content path
style = "" # style path

content_weight = 1
style_weight = 0.01

lr = 0.001
steps = 6000

generated = nst.neural_style_transfer(content,style,content_weight,style_weight,lr,steps)

generated = generated.clone().detach().cpu()
imshow(generated)
