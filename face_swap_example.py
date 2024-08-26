import time
from enum import IntEnum
import numpy as np
from xlib.face import ELandmarks2D, FLandmarks2D, FPose
from modelhub import onnx as onnx_models
import cv2
import os


tmp = onnx_models.FaceMesh.get_available_devices()
google_facemesh = onnx_models.FaceMesh(tmp[0])


face_image = cv2.imread('/home/trinity/Workspace/Deepface/DeepFaceLive/test_img.png')
lmrks = google_facemesh.extract(face_image)[0]
face_pose = FPose.from_3D_468_landmarks(lmrks)
W = face_image.shape[0]
H = face_image.shape[1]
lmrks = lmrks[...,0:2] / (W,H)
face_ulmrks = FLandmarks2D.create(ELandmarks2D.L468,lmrks)
face_ulmrks = face_ulmrks.transform(face_uni_mat, invert=True)
print('done')