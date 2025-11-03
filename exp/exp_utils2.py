# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from const_param import emo_num_dict

def intersperse(lst, item):
    """
    *I*l*v*o*y*o*u*
    Args:
        lst:
        item:

    Returns:

    """
    # Adds blank symbol
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result

def index_nointersperse(lst: torch.Tensor, rm_items):
    lst_rm = []
    lst_rm_index = []
    for item in rm_items:
        lst_rm_index.append(lst[(lst != item).nonzero()])
        lst_rm.extend(lst[(lst != item).nonzero()])
    lst_rm_index = torch.unique(torch.vstack(lst_rm_index))
    return lst_rm_index, lst_rm

def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text

def latest_checkpoint_path(dir_path, regex="grad_*.pt"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    return x

def load_checkpoint(logdir, model, num=None):
    if num is None:
        model_path = latest_checkpoint_path(logdir, regex="grad_*.pt")
    else:
        model_path = os.path.join(logdir, f"grad_{num}.pt")
    print(f'Loading checkpoint {model_path}...')
    model_dict = torch.load(model_path, map_location=lambda loc, storage: loc)
    model.load_state_dict(model_dict, strict=False)
    return model

def save_figure_to_numpy(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def plot_tensor(tensor):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data

def save_plot(tensor, savepath, size=(12, 3)):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(savepath)
    plt.close()
    return

##### For inference
def get_emo_label(emo: str,
                  emo_num_dict: dict=emo_num_dict,
                  emo_type="index",
                  emo_value=1,
                  emo_num=5
                  ):
    emo_tensor = torch.tensor([emo_num_dict[emo]], dtype=torch.long).cuda()
    if emo_type == "onehot":
        emo_tensor = torch.nn.functional.one_hot(emo_tensor, num_classes=emo_num).cuda()
    """
    emo_emb_dim = len(emo_num_dict.keys())
    emo_label = [[0] * emo_emb_dim]
    emo_label[0][emo_num_dict[emo]] = emo_value
    """

    return emo_tensor

