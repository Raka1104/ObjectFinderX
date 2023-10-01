from Detector import *
import os

def main():

# for webcam usage change to 0
    videoPath = "/Users/rakshitayadav/Raka/object_detection/test_videos/y2mate.is - 4K Road traffic video for object detection and tracking - free download now!-MNn9qKG2UFI-720p-1696088635.mp4"
    
    configPath = os.path.join("model_data", "ssd_mobilenet_v3_large_coco.pbtxt")
    modelPath = os.path.join("model_data", "frozen_inference_graph.pb")
    classesPath = os.path.join("model_data", "coco.names")

    detector = Detector(videoPath, configPath, modelPath, classesPath)
    detector.onVideo()

if __name__ == '__main__':
    main()