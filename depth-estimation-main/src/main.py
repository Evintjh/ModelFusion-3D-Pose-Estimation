#!/usr/bin/python3
"""
# > Script for inferencing UDepth on image/folder/video data
#    - Paper: https://arxiv.org/pdf/2209.12358.pdf
"""

from abc import abstractmethod
import os
import cv2
import torch
import numpy as np
import rospy
import rospkg
from sensor_msgs.msg import Image as ImageMsg
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import torchvision.transforms as transforms
import warnings

# local libs
from utils.udepth.model.udepth import *
from utils.udepth.utils.data import *
from utils.udepth.utils.utils import *
from utils.udepth.CPD.sod_mask import get_sod_mask
from utils.udepth.CPD.CPD_ResNet_models import CPD_ResNet
from utils.uw_depth.depth_estimation.utils.visualization import gray_to_heatmap

rospack = rospkg.RosPack()
PROJECT_DIR = rospack.get_path("depth")
DEPTH_TYPE = rospy.get_param("depth_type")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEPTH_TYPE == "udepth":
    # local libs
    from utils.udepth.model.udepth import *
    from utils.udepth.utils.data import *
    from utils.udepth.utils.utils import *
    from utils.udepth.CPD.sod_mask import get_sod_mask
    from utils.udepth.CPD.CPD_ResNet_models import CPD_ResNet
    UDEPTH_MODEL = os.path.join(PROJECT_DIR, "saved_models/model_RMI.pth")
    UDEPTH_CPD_MODEL = os.path.join(PROJECT_DIR, "cpd_model/CPD-R.pth")
    # Load specific model
    net = UDepth.build(n_bins=80, min_val=0.001, max_val=1, norm="linear")
    net.load_state_dict(torch.load(UDEPTH_MODEL, map_location=device, weights_only=False))
    net = net.to(device=device)
    net.eval()
    CPD_model = CPD_ResNet()
    CPD_model.load_state_dict(torch.load(UDEPTH_CPD_MODEL))
    CPD_model = CPD_model.to(device=device)
    CPD_model.eval()

elif DEPTH_TYPE == "uw_depth":
    from utils.uw_depth.depth_estimation.model.model import UDFNet
    from utils.uw_depth.depth_estimation.utils.visualization import gray_to_heatmap
    UW_DEPTH_MODEL = os.path.join(PROJECT_DIR, "saved_models/model_nullpriors.pth")
    net = UDFNet(n_bins=80).to(device)
    net.load_state_dict(torch.load(UW_DEPTH_MODEL, map_location=device, weights_only=False))
    net.eval()

else:
    raise AssertionError("Depth type undefined, supported types: udepth, uw_depth")

transform = transforms.ToTensor()
cv_bridge = CvBridge()


class Depth:
    def __init__(self) -> None:
        self.depth_enable = True
        self.depth_semaphore = 0

    @abstractmethod
    def inference(self, img):
        return img

    def image_callback(self, data):
        if self.depth_semaphore > 0:
            return
        if self.depth_enable:
            # start = timeit.default_timer()
            self.depth_semaphore += 1
            img = cv_bridge.imgmsg_to_cv2(data, "rgb8")
            img = cv2.resize(img, (640, 480))
            img = transform(img).to(device)
            result = self.inference(img)
            # result = cv_bridge.cv2_to_imgmsg(result, "8UC1")
            # print("map ", result)
            print(result.max())

            # Convert tensor to numpy array and ensure correct shape
            rospy.loginfo(f"Raw result shape: {result.shape}")
            result = result.squeeze(0).cpu().numpy()  # Remove batch dimension
            rospy.loginfo(f"After squeeze shape: {result.shape}")
            if result.ndim == 3 and result.shape[0] == 1:  # If [1, height, width]
                result = result[0]  # Remove channel dimension
            elif result.ndim == 4 and result.shape[1] == 1:  # If [1, 1, height, width]
                result = result[0, 0]  # Remove batch and channel dimensions
            
            rospy.loginfo(f"Final result shape: {result.shape}, dtype: {result.dtype}")
            result_resized = cv2.resize(result, (640, 360), interpolation=cv2.INTER_LINEAR)
            result = cv_bridge.cv2_to_imgmsg(result_resized, "32FC1")
            # result = cv_bridge.cv2_to_imgmsg(result, "32FC1")
            # print("map ", result)

            depth_pub.publish(result)
            self.depth_semaphore -= 1
            # rospy.loginfo(f"response: {timeit.default_timer() - start}")

    def toggle(self, data):
        self.depth_enable = data.data

class U_Depth(Depth):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def inference(self, img):
        """Generate depth map"""
        # Prepare SOD mask
        img = img.unsqueeze(0)
        mask = get_sod_mask(img, CPD_model).to(device)
        # Convert RGB color space into RMI input space if needed
        img = RGB_to_RMI_tensor(img)
        img = torch.autograd.Variable(img)
        # Generate depth map
        _, out = net(img)
        # # Apply guidedfilter to depth map
        # result = output_result(out, mask)
        # result = torch.mul(input=result, other=255).type(dtype=torch.uint8, non_blocking=True)
        # result = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return out


class UW_DEPTH(Depth):
    def __init__(self):
        super().__init__()
        self.prior = torch.zeros(1,2,240,320).to(device)
        self.prior[:, :, :, :] = 0.0
    
    @torch.no_grad()
    def inference(self, img):
        prediction, _ = net(img.unsqueeze(0), self.prior)
        prediction = gray_to_heatmap(prediction)
        # rospy.loginfo(f"Prediction shape: {prediction.shape}")
        prediction = prediction.squeeze(0).permute(1, 2, 0).cpu().numpy()
        prediction = (prediction * 255).astype(np.uint8)
        return prediction

if __name__ == "__main__":
    if device == "cpu":
        warnings.warn(
            f"""
            DEVICE: {device} will be very slow for depth estimation.
            Check your that your CUDA-enabled GPU is enabled
            """,
            RuntimeWarning,
            )
    rospy.init_node("depth")

    # match DEPTH_TYPE:
    #     case "udepth":
    #         depth = U_Depth()
    #     case "uw_depth":
    #         depth = UW_DEPTH()
    #     case _:
    #         raise AssertionError("Depth type undefined, supported: udepth, uw_depth")

    if DEPTH_TYPE == "udepth":
        depth = U_Depth()
    elif DEPTH_TYPE == "uw_depth":
        depth = UW_DEPTH()
    else:
        raise AssertionError("Depth type undefined, supported: udepth, uw_depth")
    rospy.Subscriber("/sensor/camera", ImageMsg, depth.image_callback, queue_size=1, buff_size=2**32)
    rospy.Subscriber("/nav/toggle_depth", Bool, depth.toggle)
    depth_pub = rospy.Publisher("/depth/map", ImageMsg, queue_size=5)
    rospy.spin()
    # rate = rospy.Rate(5)  # approx speed of inference
    # while not rospy.is_shutdown():
    #     rate.sleep()
