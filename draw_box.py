#! encoding: UTF-8

import os
from os import path
import xml.etree.cElementTree as ET 
import cv2


videos = 'carvana_video.mp4' # 视频路径
cap = cv2.VideoCapture(videos)
count = 0

output = 'carvana_video_box.mp4'  # 生成视频路径
height = 1080
width = 1616
fps = 24.01
# fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G') 用于avi格式的生成
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 用于mp4格式的生成
videowriter = cv2.VideoWriter(output, fourcc, fps, (width, height))  # 创建一个写入视频对象

while True:
    _, frame = cap.read()

    if frame is None:
        break

    bmodel_obj_bbox = list(map(int, [150/1000*width,166/1000*height,870/1000*width,766/1000*height]))
    pt_obj_bbox = list(map(int, [109/1000*width, 243/1000*height, 879/1000*width, 828/1000*height]))
    cv2.rectangle(frame, (pt_obj_bbox[0], pt_obj_bbox[1]), (pt_obj_bbox[2], pt_obj_bbox[3]), (0, 0, 255), 2)
    cv2.rectangle(frame, (bmodel_obj_bbox[0], bmodel_obj_bbox[1]), (bmodel_obj_bbox[2], bmodel_obj_bbox[3]), (0, 255, 0), 2)
    frame = frame
    count += 1

    videowriter.write(frame)

videowriter.release()
cap.release()
