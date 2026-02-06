import torch
import utmosv2
import random
import numpy as np
seed = 42
torch.manual_seed(seed)




def a():
    for a in range(4):
        # torch.manual_seed(seed)
        b()
def b():
    print(torch.randn(5))

#torch.cuda.manual_seed_all(seed)
#np.random.seed(seed)
#random.seed(seed)

# Force deterministic CuDNN ops
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    """
    for a in range(4):
        # torch.manual_seed(seed)
        print(torch.randn(1, 5)) 
    a()
    """
    import math
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    attn = torch.tensor([[
        [0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ],
        [[1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        ]], dtype=torch.int64)
    uv_mask = torch.tensor([
        [1, 0, 1, 1, 1, 1],
        [1, 0, 1, 0, 1, 1],
        ])

    cut_uv_mask_gd = torch.tensor([[
        0, 0, 1, 1, 0, 0
    ]])

    tgt_lengths_phn = (attn.sum(dim=-1) > 0).sum(dim=-1)
    #print(tgt_lengths_phn)
    import torch.nn as nn
    trd_bins = nn.Parameter(torch.linspace(torch.tensor(50), torch.tensor(600), 32 - 1))
    print(trd_bins[0], trd_bins[-1])


    """

    """

    b = np.random.randn(10)
    """
    print(b)
    print("a std: ", np.std(a))
    print("b std: ", np.std(b))

    a = ["asdf", "sdf"]
    print("".join(a))
    """