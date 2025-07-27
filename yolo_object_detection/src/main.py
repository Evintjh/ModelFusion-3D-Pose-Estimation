#!/usr/bin/python3
import rospy
from yolo_detection_class import YoloDetector

if __name__ == '__main__':

	rospy.init_node("yolo_node", anonymous=True)
	image_subscriber = YoloDetector()

	try:
		rospy.spin()
	except KeyboardInterrupt:
		pass