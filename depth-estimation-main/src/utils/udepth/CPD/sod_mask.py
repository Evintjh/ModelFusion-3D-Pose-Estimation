import torch.nn.functional as F
import torchvision.transforms as transforms
import torch


def get_sod_mask(image, cpd_model, shape=None):
    transform = transforms.Compose(
        [
            transforms.Resize((352, 352)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    _, res = cpd_model(image)
    if shape:
        res = F.interpolate(res, size=shape, mode="bilinear", align_corners=False)
    else:
        res = F.interpolate(res, size=(240, 320), mode="bilinear", align_corners=False)
    res = res.sigmoid()
    res = torch.divide(torch.subtract(res, res.min()), (res.max() - res.min() + 1e-8))

    return res
