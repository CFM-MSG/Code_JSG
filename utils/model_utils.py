
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


class RNNEncoder(nn.Module):
    """A RNN wrapper handles variable length inputs, always set batch_first=True.
    Supports LSTM, GRU and RNN. Tested with PyTorch 0.3 and 0.4
    """
    def __init__(self, word_embedding_size, hidden_size, bidirectional=True,
                 dropout_p=0, n_layers=1, rnn_type="lstm",
                 return_hidden=True, return_outputs=True,
                 allow_zero=False):
        super(RNNEncoder, self).__init__()
        """  
        :param word_embedding_size: rnn input size
        :param hidden_size: rnn output size
        :param dropout_p: between rnn layers, only useful when n_layer >= 2
        """
        self.allow_zero = allow_zero
        self.rnn_type = rnn_type
        self.n_dirs = 2 if bidirectional else 1
        # - add return_hidden keyword arg to reduce computation if hidden is not needed.
        self.return_hidden = return_hidden
        self.return_outputs = return_outputs
        self.rnn = getattr(nn, rnn_type.upper())(word_embedding_size, hidden_size, n_layers,
                                                 batch_first=True,
                                                 bidirectional=bidirectional,
                                                 dropout=dropout_p)

    def sort_batch(self, seq, lengths):
        sorted_lengths, perm_idx = lengths.sort(0, descending=True)
        if self.allow_zero:  # deal with zero by change it to one.
            sorted_lengths[sorted_lengths == 0] = 1
        reverse_indices = [0] * len(perm_idx)
        for i in range(len(perm_idx)):
            reverse_indices[perm_idx[i]] = i
        sorted_seq = seq[perm_idx]
        return sorted_seq, list(sorted_lengths), reverse_indices

    def forward(self, inputs, lengths):
        """
        inputs, sorted_inputs -> (B, T, D)
        lengths -> (B, )
        outputs -> (B, T, n_dirs * D)
        hidden -> (n_layers * n_dirs, B, D) -> (B, n_dirs * D)  keep the last layer
        - add total_length in pad_packed_sequence for compatiblity with nn.DataParallel, --remove it
        """
        assert len(inputs) == len(lengths)
        sorted_inputs, sorted_lengths, reverse_indices = self.sort_batch(inputs, lengths)
        packed_inputs = pack_padded_sequence(sorted_inputs, sorted_lengths, batch_first=True)
        outputs, hidden = self.rnn(packed_inputs)
        if self.return_outputs:
            # outputs, lengths = pad_packed_sequence(outputs, batch_first=True, total_length=int(max(lengths)))
            outputs, lengths = pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[reverse_indices]
        else:
            outputs = None
        if self.return_hidden:  #
            if self.rnn_type.lower() == "lstm":
                hidden = hidden[0]
            hidden = hidden[-self.n_dirs:, :, :]
            hidden = hidden.transpose(0, 1).contiguous()
            hidden = hidden.view(hidden.size(0), -1)
            hidden = hidden[reverse_indices]
        else:
            hidden = None
        return outputs, hidden


def pool_across_time(outputs, lengths, pool_type="max"):
    """ Get maximum responses from RNN outputs along time axis
    :param outputs: (B, T, D)
    :param lengths: (B, )
    :param pool_type: str, 'max' or 'mean'
    :return: (B, D)
    """
    if pool_type == "max":
        outputs = [outputs[i, :int(lengths[i]), :].max(dim=0)[0] for i in range(len(lengths))]
    elif pool_type == "mean":
        outputs = [outputs[i, :int(lengths[i]), :].mean(dim=0) for i in range(len(lengths))]
    else:
        raise NotImplementedError("Only support mean and max pooling")
    return torch.stack(outputs, dim=0)


def count_parameters(model, verbose=True):
    """Count number of parameters in PyTorch model,
    References: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7.

    from utils.utils import count_parameters
    count_parameters(model)
    import sys
    sys.exit(1)
    """
    n_all = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print("Parameter Count: all {:,d}; trainable {:,d}".format(n_all, n_trainable))
    return n_all, n_trainable

##############################################################
# added
def get_center_based_props(num_gauss_center, num_gauss_width, width_lower_bound=0.05, width_upper_bound=1.0):
    # lb = 1 / num_gauss_center / 2
    # gauss_center = torch.linspace(lb, 1 - lb , steps=num_gauss_center) # new
    gauss_center = torch.linspace(width_lower_bound/2, 1 - width_lower_bound/2, steps=num_gauss_center)
    # gauss_center = torch.linspace(0, 1 , steps=num_gauss_center)
    gauss_center = gauss_center.unsqueeze(-1).expand(-1, num_gauss_width).reshape(-1)
    gauss_width = torch.linspace(width_lower_bound, width_upper_bound, steps=num_gauss_width).unsqueeze(0).expand(num_gauss_center, -1).reshape(-1)

    return gauss_center, gauss_width

def get_sliding_window_based_props(map_size, step=1, width_lower_bound=0.05, width_upper_bound=1.0):
    '''
    input:
        map_size: int, the assumed length of the sequence
    '''
    centers = []
    widths = []
    # off_set = 1 / map_size / 2
    for i in range(1, map_size + 1, step): # 从2开始呢？
        count = map_size - i + 1
        lower_bound = max(i / map_size / 2 , 0) # - off_set
        upper_bound = 1 - lower_bound # + 2*off_set
        temp = np.linspace(lower_bound, upper_bound, count, endpoint=True).tolist()
        centers.extend(temp)
        
        width = i / map_size
        width = min(width_upper_bound, max(width, width_lower_bound)) # clamp width
        widths.extend([width] * count)


    gauss_center = torch.clamp(torch.tensor(centers), min=0, max=1).type(torch.float32)
    gauss_width = torch.tensor(widths).type(torch.float32)

    return gauss_center, gauss_width


def generate_gauss_weight_old(props_len, center, width, sigma=9):
    # pdb.set_trace()
    weight = torch.linspace(0, 1, props_len) # ori version, not consider the bias of position
    # lb = 1 / props_len / 2 # lower bound(relative position of the first clip's center)
    # weight = torch.linspace(lb, 1-lb, props_len) # new version TODO: test

    weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
    center = center.unsqueeze(-1)
    width = width.unsqueeze(-1).clamp(1e-2) / sigma

    w = 0.3989422804014327
    weight = w/width*torch.exp(-(weight-center)**2/(2*width**2))

    return weight/weight.max(dim=-1, keepdim=True)[0] # (center.shape[0], props_len)

# with traunc optional
def generate_gauss_weight(props_len, center, width, sigma=9, truncate=False):
    weight = torch.linspace(0, 1, props_len) # ori version, not consider the bias of position
    # lb = 1 / props_len / 2 # lower bound(relative position of the first clip's center)
    # weight = torch.linspace(lb, 1-lb, props_len) # new version TODO: test

    weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
    center = center.unsqueeze(-1)
    width = width.unsqueeze(-1).clamp(1e-2) / sigma

    w = 0.3989422804014327
    weight = w/width*torch.exp(-(weight-center)**2/(2*width**2))

    if truncate:
        # truncate
        left = ((center - width / 2) * props_len).clamp(min=0)
        left = left.type(torch.long)
        right = ((center + width / 2) * props_len).clamp(max=props_len-1)
        right = right.type(torch.long)
        for idx, w in enumerate(weight):
            weight[idx][:left[idx]] = 0
            weight[idx][right[idx]+1:] = 0

    return weight/weight.max(dim=-1, keepdim=True)[0]

def generate_mean_weight(props_len, center, width):
    weight = torch.ones((center.shape[0], props_len)).type_as(center)
    left = ((center - width / 2) * props_len).clamp(min=0)
    left = left.type(torch.long)
    right = ((center + width / 2) * props_len).clamp(max=props_len-1)
    right = right.type(torch.long)
    for idx, w in enumerate(weight):
        weight[idx][:left[idx]] = 0
        weight[idx][right[idx]+1:] = 0
    
    return weight/weight.max(dim=-1, keepdim=True)[0]

def get_gauss_props_from_clip_indices_loop(indices, num_gauss_center, num_gauss_width): # old version, slow
    row, col = indices.shape
    centers = torch.linspace(0, 1, steps=num_gauss_center)
    widthes = torch.linspace(0.05, 1.0, steps=num_gauss_width)
    gauss_center = torch.zeros_like(indices).type(torch.float32)
    gauss_width = torch.zeros_like(indices).type(torch.float32)
    for i in range(row):
        for j in range(col):
            idx = indices[i, j].item()
            center_idx, width_idx = np.unravel_index(idx, shape=(num_gauss_center, num_gauss_width))
            gauss_center[i, j], gauss_width[i, j] = centers[center_idx], widthes[width_idx]
    
    return gauss_center, gauss_width

def get_center_from_center_indices(center_indices, num_gauss_center, width_lower_bound=0.05):
    centers = torch.linspace(width_lower_bound/2, 1 - width_lower_bound/2, steps=num_gauss_center).type(torch.float32).to(center_indices.device)
    shape = center_indices.shape
    if len(shape) == 1:
        centers = centers.unsqueeze(0).expand(shape[0], -1)
    elif len(shape) == 2:
        centers = centers.unsqueeze(0).unsqueeze(0).expand(shape[0], shape[1], -1)
    elif len(shape) == 3:
        centers = centers.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(shape[0], shape[1], shape[2], -1)
    elif len(shape) == 4:
        centers = centers.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(shape[0], shape[1], shape[2], shape[3], -1)
    else:
        raise NotImplementedError("Only support 1, 2, 3, 4 dimensions")
    
    return centers.gather(-1, center_indices.unsqueeze(-1)).squeeze(-1)

def get_width_from_width_indices(width_indices, num_width_center, width_lower_bound=0.05, width_upper_bound=1):
    widthes = torch.linspace(width_lower_bound, width_upper_bound, steps=num_width_center).type(torch.float32).to(width_indices.device)
    shape = width_indices.shape
    if len(shape) == 1:
        widthes = widthes.unsqueeze(0).expand(shape[0], -1)
    elif len(shape) == 2:
        widthes = widthes.unsqueeze(0).unsqueeze(0).expand(shape[0], shape[1], -1)
    elif len(shape) == 3:
        widthes = widthes.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(shape[0], shape[1], shape[2], -1)
    elif len(shape) == 4:
        widthes = widthes.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(shape[0], shape[1], shape[2], shape[3], -1)
    else:
        raise NotImplementedError("Only support 1, 2, 3, 4 dimensions")
    
    return widthes.gather(-1, width_indices.unsqueeze(-1)).squeeze(-1)

def get_gauss_props_from_clip_indices(indices, num_gauss_center, num_gauss_width, width_lower_bound=0.05, width_upper_bound=1):
    row, col = indices.shape
    centers = torch.linspace(width_lower_bound/2, 1 - width_lower_bound/2, steps=num_gauss_center).unsqueeze(0).unsqueeze(0).expand(row, col, num_gauss_center).type(torch.float32).to(indices.device)
    widthes = torch.linspace(width_lower_bound, width_upper_bound, steps=num_gauss_width).unsqueeze(0).unsqueeze(0).expand(row, col, num_gauss_width).type(torch.float32).to(indices.device)
    # gauss_center = torch.zeros_like(indices).type(torch.float32)
    # gauss_width = torch.zeros_like(indices).type(torch.float32)

    center_indices = indices // num_gauss_width
    width_indices = indices % num_gauss_width
    gauss_center = torch.gather(centers, dim=-1, index=center_indices.unsqueeze(-1)).squeeze(-1)
    gauss_width = torch.gather(widthes, dim=-1, index=width_indices.unsqueeze(-1)).squeeze(-1)

    return gauss_center, gauss_width

# new version
def get_props_from_indices(indices, gauss_center, gauss_width):
    '''
    input:
        indices: [row, col]
        gauss_center: [num_gauss_center]
        gauss_width: [num_gauss_width]
    output:
        center_prop: [row, col]
        width_prop: [row, col]
    '''
    row, col = indices.shape

    gauss_center = gauss_center.to(indices.device)
    gauss_width = gauss_width.to(indices.device)

    expanded_centers = gauss_center.unsqueeze(0).unsqueeze(0).expand(row, col, -1)
    expanded_widths = gauss_width.unsqueeze(0).unsqueeze(0).expand(row, col, -1)

    center_prop = torch.gather(expanded_centers, dim=-1, index=indices.unsqueeze(-1)).squeeze(-1)
    width_prop = torch.gather(expanded_widths, dim=-1, index=indices.unsqueeze(-1)).squeeze(-1)

    return center_prop, width_prop

