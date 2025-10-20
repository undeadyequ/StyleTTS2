import torch
torch.set_printoptions(threshold=10_000)

def make_guided_attention_masks(ilens, olens, chunksize, sigma=0.4):
    n_batches = len(ilens)
    max_ilen = int(max(ilens))
    max_olen = int(max(olens))
    guided_attn_masks = torch.zeros((n_batches, max_ilen, max_olen)).cuda()
    for idx, (ilen, olen) in enumerate(zip(ilens, olens)):
        ilen = int(ilen)
        olen = int(olen)
        guided_attn_masks[idx, :ilen, :olen] = make_guided_attention_mask(ilen, olen, sigma, chunksize)
    return guided_attn_masks

def make_guided_attention_masks2(ilens, olens, base_sigma=0.2, ref_len=200, eps=1e-5):
    n_batches = len(ilens)
    max_ilen = int(max(ilens))
    max_olen = int(max(olens))
    guided_attn_masks = torch.zeros((n_batches, max_ilen, max_olen)).cuda()
    for idx, (ilen, olen) in enumerate(zip(ilens, olens)):
        ilen = int(ilen)
        olen = int(olen)
        guided_attn_masks[idx, :ilen, :olen] = make_gaussian_mask(ilen, olen, base_sigma, ref_len, eps)
    return guided_attn_masks


def make_gaussian_mask(T, S, base_sigma=0.2, ref_len=200, eps=1e-5):
    """
    Create a soft diagonal alignment mask between T and S with small non-zero diagonal values.

    Args:
        T (int): Length of target sequence (e.g., mel frames)
        S (int): Length of source sequence (e.g., pitch/prosody frames)
        base_sigma (float): Base std dev for the Gaussian width
        ref_len (int): Reference length for scaling sigma dynamically
        eps (float): Minimum floor value to avoid zero

    Returns:
        mask: Tensor of shape [T, S], soft diagonal Gaussian-like mask
    """
    # Dynamic sigma based on length
    sigma = base_sigma * (min(T, S) / ref_len)

    # Compute time indices
    t_idx = torch.arange(T).unsqueeze(1).float().cuda()  # [T, 1]
    s_idx = torch.arange(S).unsqueeze(0).float().cuda()  # [1, S]

    # Linear alignment path: s ≈ α * t
    #alpha = S / T
    #center = alpha * t_idx  # [T, 1]

    # Gaussian distances
    mask = 1 - torch.exp(-(t_idx/T - s_idx/S) ** 2 / (2 * sigma ** 2))  # Gaussian decay

    # Normalize along source axis (S) and clip
    #mask = mask / (mask.sum(dim=1, keepdim=True) + eps)
    mask = mask.clamp(min=eps)  # Avoid exact 0

    return mask  # [T, S]


def make_guided_attention_mask(ilen, olen, sigma, chunksize=6):
    """Make guided attention mask.
    Examples:
        #>>> guided_attn_mask =_make_guided_attention(5, 5, 0.4)
        #>>> guided_attn_mask.shape
        torch.Size([5, 5])
        #>>> guided_attn_mask
        tensor([[0.0000, 0.1175, 0.3935, 0.6753, 0.8647],
                [0.1175, 0.0000, 0.1175, 0.3935, 0.6753],
                [0.3935, 0.1175, 0.0000, 0.1175, 0.3935],
                [0.6753, 0.3935, 0.1175, 0.0000, 0.1175],
                [0.8647, 0.6753, 0.3935, 0.1175, 0.0000]])
        #>>> guided_attn_mask =_make_guided_attention(3, 6, 0.4)
        #>>> guided_attn_mask.shape
        torch.Size([6, 3])
        #>>> guided_attn_mask
        tensor([[0.0000, 0.2934, 0.7506],
                [0.0831, 0.0831, 0.5422],
                [0.2934, 0.0000, 0.2934],
                [0.5422, 0.0831, 0.0831],
                [0.7506, 0.2934, 0.0000],
                [0.8858, 0.5422, 0.0831]])
    """
    grid_x, grid_y = torch.meshgrid(torch.arange(ilen), torch.arange(olen))
    grid_x, grid_y = grid_x.float(), grid_y.float()
    #guide_mask = 1.0 - torch.exp(-(abs(grid_x / ilen - grid_y / olen) - chunksize) ** 2 / (2 * (sigma ** 2)))
    guide_mask = 1.0 - torch.exp(-(grid_x / ilen - grid_y / olen) ** 2 / (2 * (sigma ** 2)))
    if chunksize > 0:
        for id, row in enumerate(guide_mask):
            min_value, min_index = torch.min(row).item(), torch.argmin(row).item()
            guide_mask[id, max(min_index - chunksize, 0): min(min_index + chunksize, len(row))] = min_value
    return guide_mask


def make_pad_mask(lengths, xs=None, length_dim=-1):
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if xs is None:
        maxlen = int(max(lengths))
    else:
        maxlen = xs.size(length_dim)

    seq_range = torch.arange(0, maxlen, dtype=torch.int64).cuda()
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


if __name__ == '__main__':
    from utils.vis import save_plot
    ilens = torch.tensor([292]).unsqueeze(0)
    olens = torch.tensor([292]).unsqueeze(0)
    guide_matrix = make_guided_attention_masks(ilens=ilens, olens=olens, chunksize=10, sigma=0.2)  # (b, ilens_max, olens_max)
    print(guide_matrix.shape, torch.min(guide_matrix))
    save_plot(guide_matrix[0].detach().cpu(), f'./monoMask_test.png')

    guide_matrix2 = make_guided_attention_masks2(ilens, olens, base_sigma=0.2, ref_len=200, eps=0.02)
    print(guide_matrix2.shape, guide_matrix2.device, torch.min(guide_matrix2))
    save_plot(guide_matrix2[0].detach().cpu(), f'./monoMask_test2.png')

