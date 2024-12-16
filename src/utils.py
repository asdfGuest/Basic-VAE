import torch as th
import matplotlib.pyplot as plt
from typing import Tuple

def display_image(img:th.Tensor, grid_shape:Tuple[int,int], is_channel_last:bool=True) :
    '''
    Args:
        img Tensor: (n_batch, C, H, W) or (n_batch, H, W, C)
    '''
    if is_channel_last :
        img = img.permute(0, 2, 3, 1)

    fig, axes = plt.subplots(*grid_shape)

    for i, ax in enumerate(axes.flat):
        ax.imshow(img[i])
        ax.axis('off')

    plt.tight_layout()
    plt.show()
