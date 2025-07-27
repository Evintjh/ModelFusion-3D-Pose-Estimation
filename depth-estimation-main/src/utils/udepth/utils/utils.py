"""
Functions of average meter, logger, guidedfilter, and result output
"""
import logging
from cv2.ximgproc import guidedFilter
import numpy as np
import torch as T
import torch.nn as nn
from torch.nn.functional import normalize
import rospy


def box_filter_1d(tensor, dim, r):
    cs = tensor.cumsum(dim).transpose(dim, 0)
    return T.cat([  # left side, center, right side
        cs[r: 2*r+1],
        cs[2*r+1:] - cs[:-2*r-1],
        cs[-1:] - cs[-2*r-1: -r-1]]
        ).transpose(dim, 0)

class AverageMeter(object):
    """Compute and store the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class BoxFilterND(nn.Module):
    """Compute a fast sum filter with a square window of side length 2*radius
    over the given dimension.  (ie equivalent result to convolution with kernel
    of all ones, but much faster).  At edges, behave as if padding zeros
    (equivalent to mode='constant' with a fill value of 0).
    Makes use of the fact that summation is separable along each dimension.
    This is adapted from the matlab code provided by Kaiming He, and
    generalized to any dims.
    """
    def __init__(self, radius, dims):
        super().__init__()
        self.dims = dims
        self.radius = radius

    def forward(self, tensor):
        for dim in self.dims:
            assert tensor.shape[dim] > 2*self.radius, \
                    "BoxFilter: all dimensions must be larger than radius"
            tensor = box_filter_1d(tensor, dim, self.radius)
        return tensor
 
class GuidedFilterND(nn.Module):
    """PyTorch GuidedFilter for a multi-channel guide image and a 1 channel
    source image.
    See Section 3.5 of the 2013 Guided Filter paper by Kaiming He et. al,
    and also Algorithm 2 on arXiv https://arxiv.org/pdf/1505.00996.pdf
    For the Fast Guided Filter, pass either a subsampled filter image `p` when
    calling the forward method, or at initialization, pass a subsampling_ratio
    >=1 to subsample the image before computations. (ie. a value of 2 samples
    every other pixel).  This makes the algorithm faster on large images with
    little loss in detail.  By default, this implementation will try
    to infer if the filter image p has been downsampled.  An error is raised if
    you both pass in a p that is a different shape than I and also pass in a
    subsampling ratio.
    Note: `radius` and `subsampling_ratio` are not differentiable, but
    `eps` is differentiable and could be torch.Tensor(eps, requires_grad=True)
    """
    def __init__(self, radius: int, eps: float, subsampling_ratio: int = 1):
        super().__init__()
        self.subsampling_ratio = subsampling_ratio
        self.radius = radius
        self.eps = eps

    def forward(self, I, p):
        """
        - I is the guide image (3,4, or 5 dimensional image),
        where first two dims are the (batch_size, channels, h,w,extra,extra)
        - p is the filter image (batch_size, c', ...)
        where c' satisfies c' <= channels (typically c'=1 or c'=channels)
        """
        ndim = I.dim() - 2  # for scale factor
        # determine if fast guided filter (ie are we using downsampling?)
        if p.shape[-1] != I.shape[-1]:
            is_fast = True
            # infer the subsampling ratio for fast guided filter
            subsampling_ratio = I.shape[-1] / p.shape[-1]
            I_orig = I
            I = T.nn.functional.interpolate(
                I, size=p.shape[2:], mode='bilinear')
            radius = round(self.radius / subsampling_ratio)
            if self.subsampling_ratio != 1:
                raise Exception(
                    f"{self.__class__.__name__}: either the filter img p must"
                    " be same size as I, or don't pass a subsampling_ratio")
        elif self.subsampling_ratio != 1:
            is_fast = True
            # fast guided filter with a predefined subsampling ratio
            I_orig = I
            scale_factor = (1/self.subsampling_ratio, ) * ndim
            I = T.nn.functional.interpolate(
                I, scale_factor=scale_factor, mode='bilinear')
            p = T.nn.functional.interpolate(
                p, scale_factor=scale_factor, mode='bilinear')
            radius = round(self.radius / self.subsampling_ratio)
        else:
            is_fast = False
            radius = self.radius
        # now do the guided filter operations
        bs,c = I.shape[:2]
        _I_shape2 = I.shape[2:]
        _I_dims = list(range(I.dim()))[2:]
        # --> assign letter for each dimension of the image
        hw = ''.join(einsum_letter for einsum_letter in 'hwzyx'[:I.dim()-2])

        f = BoxFilterND(radius, dims=range(2, I.dim()))
        N = f(T.ones_like(I[:,[0]]))

        I_mean = f(I) / N
        p_mean = f(p) / N

        Ip_mean = f(p * I) / N
        first_term = (Ip_mean - p_mean * I_mean)

        _cov = T.einsum(f'bc{hw},bd{hw}->bcd{hw}', I, I)\
                .reshape(bs, c*c, *_I_shape2)
        cov = (f(_cov) / N).reshape(bs, c, c, *_I_shape2)\
                .permute(0, *(x+1 for x in _I_dims), 1, 2)
        eps_mat = self.eps * T.eye(c).reshape(1, *[1 for _ in _I_dims], c, c)
        eps_mat = eps_mat.to(device="cuda")
        second_term = T.inverse(cov + eps_mat)

        A = T.einsum(f'bc{hw},b{hw}cd->bc{hw}', first_term, second_term)
        b = p_mean - T.einsum(f'bc{hw},bd{hw}->b{hw}', A, I_mean).unsqueeze_(1)

        A_mean = f(A) / N
        b_mean = f(b) / N

        if is_fast:
            I = I_orig
            A_mean = T.nn.functional.interpolate(
                A_mean, size=I.shape[2:], mode='bilinear')
            b_mean = T.nn.functional.interpolate(
                b_mean, size=I.shape[2:], mode='bilinear')

        q = T.einsum(f'bc{hw},bd{hw}->b{hw}', A_mean, I).unsqueeze_(1) + b_mean
        return q

def get_logger(filename, verbosity=1, name=None):
    """Get logger"""
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger


def gf(mask, img):
    """Apply guidedfilter"""
    #radius, eps, dDepth
    return guidedFilter(mask, img, 4, 0.2, -1)

gfND = GuidedFilterND(radius=4, eps=0.1)

def output_result(out, mask):
    """Prepare data and apply guidedfilter to depth map"""
    # for numpy implementation:
    """
    mask = np.float32(mask)
    max_item = max(max(row) for row in img_resized)
    img_resized = img_resized / max_item * 255
    rospy.loginfo(f"img: {img_resized}, mask: {mask}")
    result = gf(mask, img_resized)
    """
    out = out.reshape(1, 1, 240, 320)
    # out = T.mul(input=out, other=255)
    # mask = T.mul(input=mask, other=255)
    result = gfND.forward(mask, out)

    return result

