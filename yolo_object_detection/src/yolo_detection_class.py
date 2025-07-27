# #!/usr/bin/python3



# import rospy
# import cv2
# import signal
# import torch
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image
# from ultralytics import YOLO
# from torch.backends import cudnn
# from yolo_object_detection.msg import YoloInference, BoundingBox
# import actionlib
# from yolo_object_detection.msg import OnYoloAction, OnYoloGoal, OnYoloResult

# class YoloDetector:
	
# 	def __init__(self):		
# 		self.is_yolo_ready = rospy.get_param('~default', False)
# 		if self.is_yolo_ready:
# 			rospy.logwarn("YOLO of node %s is set to on by default. Getting ready now.." % rospy.get_name())
# 			self.intialize_node()
				
#         # Action Server to activate YOLO
# 		server_name = rospy.get_name() + "/activate_yolo_action"
# 		self.action_server = actionlib.SimpleActionServer(server_name, OnYoloAction, self.on_yolo_callback, False)
# 		self.action_server.start()		
# 		signal.signal(signal.SIGINT, self.signal_handler)


# 	def intialize_node(self):
# 		yolo_model_name = rospy.get_param('~yolo_model_name')
# 		input_image_topic = rospy.get_param('~input_image_topic')
# 		inference_topic = rospy.get_param('~inference_topic')
# 		annotated_frame_topic = rospy.get_param('~annotated_frame_topic')

# 		try:
# 			self.model = YOLO(yolo_model_name)
# 			rospy.loginfo("Using %s for inference" % yolo_model_name)
# 		except:
# 			rospy.loginfo("Cannot load %s" % yolo_model_name)
# 			return False
# 		# cudnn.benchmark = True

# 		self.bridge = CvBridge()
# 		self.annotated_frame_publisher = rospy.Publisher(annotated_frame_topic, Image, queue_size=10)
# 		self.inference_publisher = rospy.Publisher(inference_topic, YoloInference, queue_size=10)
# 		rospy.Subscriber(input_image_topic, Image, self.inference_callback)
		
# 		return True
		

# 	def on_yolo_callback(self, goal: OnYoloGoal):
# 		result = OnYoloResult()

# 		if not goal.turn_on_or_off:
# 			rospy.logwarn("Turning off YOLO and kill this node.")
# 			self.cleanup()
# 			self.is_yolo_ready = False
# 			result.status = True
# 			rospy.logwarn("YOLO is off. Node killed!")
# 			self.action_server.set_succeeded(result)
# 			exit()

# 		if self.is_yolo_ready:
# 			rospy.logwarn("YOLO is already on.")
# 			result.status = True
# 			self.action_server.set_succeeded(result)
# 			return
		
# 		rospy.loginfo("Turning on YOLO. Expect some delay...")
# 		result.status = self.intialize_node()	# If failed, status is False	
# 		self.action_server.set_succeeded(result)
# 		if result.status:
# 			rospy.loginfo("YOLO is ready!")		
# 			self.is_yolo_ready = True
# 		else:
# 			rospy.loginfo("Failed to turn on YOLO. Please check the parameters.")
# 			self.cleanup()
# 			exit()


# 	def inference_callback(self, msg):
# 		if not self.is_yolo_ready:
# 			return
		
# 		try:
# 			frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
# 			print(frame.shape)
# 		except:
# 			return
		
# 		results = self.model(source=frame, conf=0.6, iou=0.5, verbose=False)[0]
# 		self.publish_box_inference(results)

# 		# Publish the annotated frame
# 		try:
# 			annotated_frame = results.plot()
# 			height, width = annotated_frame.shape[:2]
# 			center_x = width // 2
# 			center_y = height // 2
# 			rospy.loginfo(f"Image size: {width} x {height} pixels")
# 			rospy.loginfo(f"Center pixel: {center_x}, {center_y}")
# 			self.annotated_frame_publisher.publish(self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8"))
# 		except:
# 			pass


# 	def publish_box_inference(self, results):

# 		boxes = results.boxes
# 		xywhn = boxes.xywhn		# A 2D tensor of shape (N, 4) where N is the number of detected objects
# 		conf = boxes.conf		# A 1D tensor of shape (N,) containing the confidence scores of the detected objects
# 		cls = boxes.cls			# A 1D tensor of shape (N,) containing the class labels of the detected objects

# 		conf = conf.reshape(-1, 1)
# 		cls = cls.reshape(-1, 1)
# 		results_tensor = torch.cat((xywhn, conf, cls), dim=1)

# 		msg = YoloInference()
# 		msg.header.stamp = rospy.Time.now()
# 		msg.header.frame_id = "yolo_inference"
# 		for i in range(results_tensor.shape[0]):
# 			result = results_tensor[i]
# 			bounding_box = BoundingBox()
# 			bounding_box.x = result[0]
# 			bounding_box.y = result[1]
# 			bounding_box.w = result[2]
# 			bounding_box.h = result[3]
# 			bounding_box.conf = result[4]
# 			bounding_box.cls = result[5]
# 			msg.bounding_boxes.append(bounding_box)

# 		self.inference_publisher.publish(msg)		

# 	def signal_handler(self, signum, frame):
# 		self.cleanup()

# 	def cleanup(self):
# 		rospy.loginfo("Shutting down %s" % rospy.get_name())
# 		cv2.destroyAllWindows()




#!/usr/bin/python3

import rospy
import cv2
import signal
import torch
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ultralytics import YOLO
from torch.backends import cudnn
from yolo_object_detection.msg import YoloInference, BoundingBox
import actionlib
from yolo_object_detection.msg import OnYoloAction, OnYoloGoal, OnYoloResult
import numpy as np

class YoloDetector:
    def __init__(self):        
        self.is_yolo_ready = rospy.get_param('~default', False)
        if self.is_yolo_ready:
            rospy.logwarn("YOLO of node %s is set to on by default. Getting ready now.." % rospy.get_name())
            self.intialize_node()
                
        # Action Server to activate YOLO
        server_name = rospy.get_name() + "/activate_yolo_action"
        self.action_server = actionlib.SimpleActionServer(server_name, OnYoloAction, self.on_yolo_callback, False)
        self.action_server.start()        
        signal.signal(signal.SIGINT, self.signal_handler)

        # Initialize depth map subscriber
        self.depth_map = None
        self.depth_sub = rospy.Subscriber("/depth/map", Image, self.depth_callback, queue_size=10)

    def depth_callback(self, msg):
        """Callback to store the latest depth map"""
        self.depth_map = self.bridge.imgmsg_to_cv2(msg, "32FC1")  # depth map is 32FC1 -> converts 0-255 scale back to 0-1 original depth map range

    def intialize_node(self):
        yolo_model_name = rospy.get_param('~yolo_model_name')
        input_image_topic = rospy.get_param('~input_image_topic')
        inference_topic = rospy.get_param('~inference_topic')
        annotated_frame_topic = rospy.get_param('~annotated_frame_topic')

        try:
            self.model = YOLO(yolo_model_name)
            rospy.loginfo("Using %s for inference" % yolo_model_name)
        except:
            rospy.loginfo("Cannot load %s" % yolo_model_name)
            return False
        # cudnn.benchmark = True

        self.bridge = CvBridge()
        self.annotated_frame_publisher = rospy.Publisher(annotated_frame_topic, Image, queue_size=10)
        self.inference_publisher = rospy.Publisher(inference_topic, YoloInference, queue_size=10)
        self.depth_annotated_publisher = rospy.Publisher("/depth_annotated_frame", Image, queue_size=10)  # New topic
        rospy.Subscriber(input_image_topic, Image, self.inference_callback)
        
        return True
        

    def on_yolo_callback(self, goal: OnYoloGoal):
        result = OnYoloResult()

        if not goal.turn_on_or_off:
            rospy.logwarn("Turning off YOLO and kill this node.")
            self.cleanup()
            self.is_yolo_ready = False
            result.status = True
            rospy.logwarn("YOLO is off. Node killed!")
            self.action_server.set_succeeded(result)
            exit()

        if self.is_yolo_ready:
            rospy.logwarn("YOLO is already on.")
            result.status = True
            self.action_server.set_succeeded(result)
            return
        
        rospy.loginfo("Turning on YOLO. Expect some delay...")
        result.status = self.intialize_node()    # If failed, status is False    
        self.action_server.set_succeeded(result)
        if result.status:
            rospy.loginfo("YOLO is ready!")        
            self.is_yolo_ready = True
        else:
            rospy.loginfo("Failed to turn on YOLO. Please check the parameters.")
            self.cleanup()
            exit()


    # def inference_callback(self, msg):
    #     if not self.is_yolo_ready:
    #         return
        
    #     try:
    #         frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
    #         print(frame.shape)
    #     except:
    #         return
        
    #     results = self.model(source=frame, conf=0.6, iou=0.5, verbose=False)[0]
    #     self.publish_box_inference(results)

    #     # Publish the annotated frame
    #     try:
    #         annotated_frame = results.plot()
    #         height, width = annotated_frame.shape[:2]
    #         center_x = width // 2
    #         center_y = height // 2
    #         rospy.loginfo(f"Image size: {width} x {height} pixels")
    #         rospy.loginfo(f"Center pixel: {center_x}, {center_y}")
    #         self.annotated_frame_publisher.publish(self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8"))

    #         # Overlay depth map within bounding boxes
    #         if self.depth_map is not None:
    #             # Resize depth map to match annotated frame
    #             depth_map_resized = cv2.resize(self.depth_map, (width, height))
    #             # Convert depth map to 3-channel for overlay (grayscale to BGR)
    #             depth_map_color = cv2.cvtColor(depth_map_resized, cv2.COLOR_GRAY2BGR)

    #             # Create output image for depth overlay
    #             depth_annotated_frame = annotated_frame.copy()
    #             boxes = results.boxes.xyxy  # [x1, y1, x2, y2] coordinates

    #             for box in boxes:
    #                 x1, y1, x2, y2 = map(int, box[:4])
    #                 # Create mask for the bounding box region
    #                 mask = np.zeros((height, width), dtype=np.uint8)
    #                 mask[y1:y2, x1:x2] = 255
    #                 # Apply mask to depth map
    #                 masked_depth = cv2.bitwise_and(depth_map_color, depth_map_color, mask=mask)
    #                 # Overlay masked depth onto annotated frame
    #                 depth_annotated_frame[y1:y2, x1:x2] = cv2.addWeighted(
    #                     depth_annotated_frame[y1:y2, x1:x2], 0.7,
    #                     masked_depth[y1:y2, x1:x2], 0.3, 0.0
    #                 )

    #             # Publish the depth-annotated frame
    #             self.depth_annotated_publisher.publish(self.bridge.cv2_to_imgmsg(depth_annotated_frame, "bgr8"))

    #     except Exception as e:
    #         rospy.logerr(f"Error in overlay: {str(e)}")
    #         pass

    def inference_callback(self, msg):
        if not self.is_yolo_ready:
            return
        
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # print(frame.shape)
        except:
            return
        
        results = self.model(source=frame, conf=0.6, iou=0.5, verbose=False)[0]
        self.publish_box_inference(results)

        # Publish the annotated frame
        try:
            annotated_frame = results.plot()
            height, width = annotated_frame.shape[:2]
            center_x = width // 2
            center_y = height // 2
            rospy.loginfo(f"Image size: {width} x {height} pixels")
            rospy.loginfo(f"Center pixel: {center_x}, {center_y}")
            self.annotated_frame_publisher.publish(self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8"))

            # Overlay depth map within bounding boxes (only pixels > 200, preserving original values)
            if self.depth_map is not None:
                # Resize depth map to match annotated frame
                depth_map_resized = cv2.resize(self.depth_map, (width, height))
                # Convert depth map to 3-channel for overlay (grayscale to BGR)
                depth_map_color = cv2.cvtColor(depth_map_resized, cv2.COLOR_GRAY2BGR)

                # Create output image for depth overlay
                depth_annotated_frame = annotated_frame.copy()
                boxes = results.boxes.xyxy  # [x1, y1, x2, y2] coordinates
                
                mask_color = [0, 255, 0]  # Green (B, G, R)
                m = -9.3134
                c = 7.7555
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    # Extract the region of interest from the depth map
                    roi_depth = depth_map_resized[y1:y2, x1:x2]
                    rospy.loginfo(f"ROI depth min: {np.min(roi_depth)}, max: {np.max(roi_depth)}")
                    # print("ROI depth:", roi_depth)
                    # Calculate top 60% of the ROI height
                    roi_height = y2 - y1
                    top_60_height = int(roi_height * 0.6)
                    top_60_depth = roi_depth[:top_60_height, :]

                    # Create a mask for pixels with depth > 0.49 in the top 60%
                    threshold = 0.49  # Midpoint of 0.001-1 range
                    mask = (top_60_depth > threshold).astype(np.uint8)  # Binary mask (0 or 1)
                    rospy.loginfo(f"Mask shape: {mask.shape}, Mask min: {np.min(mask)}, max: {np.max(mask)}")

                    # Compute average of pixels above threshold in top 60%
                    if np.any(mask):
                        above_threshold_values = top_60_depth[mask == 1]
                        average_depth = np.mean(above_threshold_values)
                        # Calculate distance using linear model y = mx
                        distance = m * average_depth + c
                        rospy.loginfo(f"Average depth above threshold (0.49) in top 60% of ROI: {average_depth}, Calculated distance: {distance} meters")

                    # Create a colored mask image for the full ROI, with color applied to top 60% above threshold
                    mask_height, mask_width = y2 - y1, x2 - x1
                    mask_3d = np.zeros((mask_height, mask_width, 3), dtype=np.uint8)
                    # Extend the top 60% mask to the full height, applying color only where mask is 1
                    top_60_mask_extended = np.zeros((mask_height, mask_width), dtype=np.uint8)
                    top_60_mask_extended[:top_60_height, :] = mask
                    mask_3d[top_60_mask_extended == 1] = mask_color

                    # Overlay the colored mask onto the annotated frame
                    depth_annotated_frame[y1:y2, x1:x2] = cv2.addWeighted(
                        depth_annotated_frame[y1:y2, x1:x2], 0.7,
                        mask_3d, 0.3, 0.0
                    )

                    # Add distance text label above the bounding box
                    if np.any(mask):  # Only add text if there’s a valid average
                        text = f"Dist: {distance:.2f}m"
                        text_position = (x1 + 5, y1 + 30)  # Position above the box
                        cv2.putText(depth_annotated_frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5, (0, 0, 0), 2, cv2.LINE_AA)

                # Publish the depth-annotated frame
                self.depth_annotated_publisher.publish(self.bridge.cv2_to_imgmsg(depth_annotated_frame, "bgr8"))

        except Exception as e:
            rospy.logerr(f"Error in overlay: {str(e)}")
            pass


    def publish_box_inference(self, results):
        boxes = results.boxes
        xywhn = boxes.xywhn    # A 2D tensor of shape (N, 4) where N is the number of detected objects
        conf = boxes.conf      # A 1D tensor of shape (N,) containing the confidence scores
        cls = boxes.cls        # A 1D tensor of shape (N,) containing the class labels

        conf = conf.reshape(-1, 1)
        cls = cls.reshape(-1, 1)
        results_tensor = torch.cat((xywhn, conf, cls), dim=1)

        msg = YoloInference()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "yolo_inference"
        for i in range(results_tensor.shape[0]):
            result = results_tensor[i]
            bounding_box = BoundingBox()
            bounding_box.x = result[0]
            bounding_box.y = result[1]
            bounding_box.w = result[2]
            bounding_box.h = result[3]
            bounding_box.conf = result[4]
            bounding_box.cls = result[5]
            msg.bounding_boxes.append(bounding_box)

        self.inference_publisher.publish(msg)        

    def signal_handler(self, signum, frame):
        self.cleanup()

    def cleanup(self):
        rospy.loginfo("Shutting down %s" % rospy.get_name())
        cv2.destroyAllWindows()










#### BAD FILTERING

# import rospy
# import cv2
# import signal
# import torch
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image
# from ultralytics import YOLO
# from torch.backends import cudnn
# from yolo_object_detection.msg import YoloInference, BoundingBox
# import actionlib
# from yolo_object_detection.msg import OnYoloAction, OnYoloGoal, OnYoloResult
# from average_filter_python.ave_filter import AverageFilter

# class YoloDetector:
	
# 	def __init__(self):		
# 		self.is_yolo_ready = rospy.get_param('~default', False)
# 		if self.is_yolo_ready:
# 			rospy.logwarn("YOLO of node %s is set to on by default. Getting ready now.." % rospy.get_name())
# 			self.intialize_node()
				
#         # Action Server to activate YOLO
# 		server_name = rospy.get_name() + "/activate_yolo_action"
# 		self.action_server = actionlib.SimpleActionServer(server_name, OnYoloAction, self.on_yolo_callback, False)
# 		self.action_server.start()		
# 		self.ave_filter_x = AverageFilter(filter_size = 3)
# 		self.ave_filter_y = AverageFilter(filter_size = 3)
# 		self.ave_filter_w = AverageFilter(filter_size = 3)
# 		self.ave_filter_h = AverageFilter(filter_size = 3)
# 		self.old_result_x = self.old_result_y = self.old_result_w = 0.0
# 		signal.signal(signal.SIGINT, self.signal_handler)


# 	def intialize_node(self):
# 		yolo_model_name = rospy.get_param('~yolo_model_name')
# 		input_image_topic = rospy.get_param('~input_image_topic')
# 		inference_topic = rospy.get_param('~inference_topic')
# 		annotated_frame_topic = rospy.get_param('~annotated_frame_topic')

# 		try:
# 			self.model = YOLO(yolo_model_name)
# 			rospy.loginfo("Using %s for inference" % yolo_model_name)
# 		except:
# 			rospy.loginfo("Cannot load %s" % yolo_model_name)
# 			return False
# 		# cudnn.benchmark = True

# 		self.bridge = CvBridge()
# 		self.annotated_frame_publisher = rospy.Publisher(annotated_frame_topic, Image, queue_size=10)
# 		self.inference_publisher = rospy.Publisher(inference_topic, YoloInference, queue_size=10)
# 		rospy.Subscriber(input_image_topic, Image, self.inference_callback)
		
# 		return True
		

# 	def on_yolo_callback(self, goal: OnYoloGoal):
# 		result = OnYoloResult()

# 		if not goal.turn_on_or_off:
# 			rospy.logwarn("Turning off YOLO and kill this node.")
# 			self.cleanup()
# 			self.is_yolo_ready = False
# 			result.status = True
# 			rospy.logwarn("YOLO is off. Node killed!")
# 			self.action_server.set_succeeded(result)
# 			exit()

# 		if self.is_yolo_ready:
# 			rospy.logwarn("YOLO is already on.")
# 			result.status = True
# 			self.action_server.set_succeeded(result)
# 			return
		
# 		rospy.loginfo("Turning on YOLO. Expect some delay...")
# 		result.status = self.intialize_node()	# If failed, status is False	
# 		self.action_server.set_succeeded(result)
# 		if result.status:
# 			rospy.loginfo("YOLO is ready!")		
# 			self.is_yolo_ready = True
# 		else:
# 			rospy.loginfo("Failed to turn on YOLO. Please check the parameters.")
# 			self.cleanup()
# 			exit()


# 	def inference_callback(self, msg):
# 		if not self.is_yolo_ready:
# 			return
		
# 		try:
# 			frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
# 		except:
# 			return
		
# 		results = self.model(source=frame, conf=0.6, iou=0.5, verbose=False)[0]
# 		self.publish_box_inference(results)

# 		# Publish the annotated frame
# 		try:
# 			annotated_frame = results.plot()
# 			self.annotated_frame_publisher.publish(self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8"))
# 		except:
# 			pass


# 	def publish_box_inference(self, results):

# 		boxes = results.boxes
# 		xywhn = boxes.xywhn		# A 2D tensor of shape (N, 4) where N is the number of detected objects
# 		conf = boxes.conf		# A 1D tensor of shape (N,) containing the confidence scores of the detected objects
# 		cls = boxes.cls			# A 1D tensor of shape (N,) containing the class labels of the detected objects

# 		conf = conf.reshape(-1, 1)
# 		cls = cls.reshape(-1, 1)
# 		results_tensor = torch.cat((xywhn, conf, cls), dim=1)

# 		msg = YoloInference()
# 		msg.header.stamp = rospy.Time.now()
# 		msg.header.frame_id = "yolo_inference"
		
# 		# # Initialize or use old results
# 		# if not hasattr(self, 'old_result_x'):
# 		# 	self.old_result_x = self.old_result_y = self.old_result_w = 0.0

# 		for i in range(results_tensor.shape[0]):
# 			result = results_tensor[i]  # Assign result before using it in the condition


# 			# # Skip the result if change exceeds a threshold
# 			# if abs(result[0] - self.old_result_x) > 0.5 or \
# 			# abs(result[1] - self.old_result_y) > 0.5 or \
# 			# abs(result[2] - self.old_result_w) > 0.5:
# 			# 	continue

# 			# # Update old result values
# 			# self.old_result_x, self.old_result_y, self.old_result_w = result[0], result[1], result[2]


# 			bounding_box = BoundingBox()
# 			bounding_box.x = self.ave_filter_x.moving_avg(result[0])
# 			bounding_box.y = self.ave_filter_y.moving_avg(result[1])
# 			bounding_box.w = self.ave_filter_w.moving_avg(result[2])
# 			bounding_box.h = self.ave_filter_h.moving_avg(result[3])
# 			bounding_box.conf = result[4]
# 			bounding_box.cls = result[5]
# 			msg.bounding_boxes.append(bounding_box)

# 			# bounding_box = BoundingBox()
# 			# bounding_box.x = result[0]
# 			# bounding_box.y = result[1]
# 			# bounding_box.w = result[2]
# 			# bounding_box.h = result[3]
# 			# bounding_box.conf = result[4]
# 			# bounding_box.cls = result[5]
		
# 		self.inference_publisher.publish(msg)		

# 	def signal_handler(self, signum, frame):
# 		self.cleanup()

# 	def cleanup(self):
# 		rospy.loginfo("Shutting down %s" % rospy.get_name())
# 		cv2.destroyAllWindows()




# import rospy
# import cv2
# import signal
# import torch
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image
# from ultralytics import YOLO
# from yolo_object_detection.msg import YoloInference, BoundingBox
# import actionlib
# from yolo_object_detection.msg import OnYoloAction, OnYoloGoal, OnYoloResult
# from average_filter_python.ave_filter import AverageFilter

# class YoloDetector:
    
#     def __init__(self):        
#         self.is_yolo_ready = rospy.get_param('~default', False)
#         if self.is_yolo_ready:
#             rospy.logwarn("YOLO of node %s is set to on by default. Getting ready now.." % rospy.get_name())
#             self.initialize_node()
                
#         # Action Server to activate YOLO
#         server_name = rospy.get_name() + "/activate_yolo_action"
#         self.action_server = actionlib.SimpleActionServer(server_name, OnYoloAction, self.on_yolo_callback, False)
#         self.action_server.start()

#         # Dictionary to store filters for each detected object
#         self.object_filters = {}
#         self.bridge = None
#         self.annotated_frame_publisher = None
#         self.inference_publisher = None
#         signal.signal(signal.SIGINT, self.signal_handler)


#     def initialize_node(self):
#         yolo_model_name = rospy.get_param('~yolo_model_name')
#         input_image_topic = rospy.get_param('~input_image_topic')
#         inference_topic = rospy.get_param('~inference_topic')
#         annotated_frame_topic = rospy.get_param('~annotated_frame_topic')

#         try:
#             self.model = YOLO(yolo_model_name)
#             rospy.loginfo("Using %s for inference" % yolo_model_name)
#         except Exception as e:
#             rospy.logerr(f"Cannot load {yolo_model_name}: {e}")
#             return False

#         self.bridge = CvBridge()
#         self.annotated_frame_publisher = rospy.Publisher(annotated_frame_topic, Image, queue_size=10)
#         self.inference_publisher = rospy.Publisher(inference_topic, YoloInference, queue_size=10)
#         rospy.Subscriber(input_image_topic, Image, self.inference_callback)
        
#         return True
        

#     def on_yolo_callback(self, goal: OnYoloGoal):
#         result = OnYoloResult()

#         if not goal.turn_on_or_off:
#             rospy.logwarn("Turning off YOLO and killing this node.")
#             self.cleanup()
#             self.is_yolo_ready = False
#             result.status = True
#             rospy.logwarn("YOLO is off. Node killed!")
#             self.action_server.set_succeeded(result)
#             exit()

#         if self.is_yolo_ready:
#             rospy.logwarn("YOLO is already on.")
#             result.status = True
#             self.action_server.set_succeeded(result)
#             return
        
#         rospy.loginfo("Turning on YOLO. Expect some delay...")
#         result.status = self.initialize_node()  # If failed, status is False    
#         self.action_server.set_succeeded(result)
#         if result.status:
#             rospy.loginfo("YOLO is ready!")        
#             self.is_yolo_ready = True
#         else:
#             rospy.loginfo("Failed to turn on YOLO. Please check the parameters.")
#             self.cleanup()
#             exit()


#     def inference_callback(self, msg):
#         if not self.is_yolo_ready:
#             return
        
#         try:
#             frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
#         except:
#             return
        
#         results = self.model(source=frame, conf=0.6, iou=0.5, verbose=False)[0]
#         self.publish_box_inference(results)

#         # Publish the annotated frame
#         try:
#             annotated_frame = results.plot()
#             self.annotated_frame_publisher.publish(self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8"))
#         except:
#             pass


#     def publish_box_inference(self, results):
#         boxes = results.boxes
#         if boxes is None or len(boxes) == 0:
#             rospy.logwarn("No detections from YOLO")
#             return

#         xywhn = boxes.xywhn
#         conf = boxes.conf.reshape(-1, 1)
#         cls = boxes.cls.reshape(-1, 1)
#         results_tensor = torch.cat((xywhn, conf, cls), dim=1)

#         msg = YoloInference()
#         msg.header.stamp = rospy.Time.now()
#         msg.header.frame_id = "yolo_inference"

#         for i in range(results_tensor.shape[0]):
#             result = results_tensor[i]
#             object_key = f"{int(result[5].item())}_{round(result[0].item(), 2)}_{round(result[1].item(), 2)}"

#             # Initialize filters for this object if not already done
#             if object_key not in self.object_filters:
#                 self.object_filters[object_key] = {
#                     "x": AverageFilter(filter_size=3),
#                     "y": AverageFilter(filter_size=3),
#                     "w": AverageFilter(filter_size=3),
#                     "h": AverageFilter(filter_size=3)
#                 }

#             filters = self.object_filters[object_key]

#             # Apply averaging for this specific object
#             bounding_box = BoundingBox()
#             bounding_box.x = filters["x"].moving_avg(result[0])
#             bounding_box.y = filters["y"].moving_avg(result[1])
#             bounding_box.w = filters["w"].moving_avg(result[2])
#             bounding_box.h = filters["h"].moving_avg(result[3])
#             bounding_box.conf = result[4]
#             bounding_box.cls = result[5]

#             rospy.loginfo(f"Appending bounding box: {bounding_box}")
#             msg.bounding_boxes.append(bounding_box)

#         rospy.loginfo(f"Publishing {len(msg.bounding_boxes)} bounding boxes")
#         self.inference_publisher.publish(msg)


#     def signal_handler(self, signum, frame):
#         self.cleanup()


#     def cleanup(self):
#         rospy.loginfo("Shutting down %s" % rospy.get_name())
#         cv2.destroyAllWindows()
