import time
import torch
import numpy as np
import torchvision.transforms as transforms

from queue import Queue
from threading import Thread

from Detection.Models import Darknet
from Detection.Utils import non_max_suppression, ResizePadding


class TinyYOLOv3_onecls(object):
    """Load trained Tiny-YOLOv3 one class (person) detection model.
    Args:
        input_size: (int) Size of input image must be divisible by 32. Default: 416,
        config_file: (str) Path to Yolo model structure config file.,
        yolo 모델 구조 구성 파일의 경로
        weight_file: (str) Path to trained weights file.,
        훈련된 가중치 파일의 경로
        nms: (float) Non-Maximum Suppression overlap threshold.,
        conf_thres: (float) Minimum Confidence threshold of predicted bboxs to cut off.,
        device: (str) Device to load the model on 'cpu' or 'cuda'.
    """
    def __init__(self,
                 input_size=416,
                 config_file='Models/yolo-tiny-onecls/yolov3-tiny-onecls.cfg',  # YOLO 신경망의 layout(계층)이 설정된 파일
                 weight_file='Models/yolo-tiny-onecls/best-model.pth',
                 nms=0.2,
                 conf_thres=0.45,
                 device='cuda'):
        self.input_size = input_size # 탐색할 프레임 이미지의 size(고정값)
        self.model = Darknet(config_file).to(device) # Darknet : config_file의 구조를 분석해  YOLO 신경망 생성
        self.model.load_state_dict(torch.load(weight_file)) 
        self.model.eval()
        self.device = device

        self.nms = nms # 동일한 이미지에서 다수의 검출 문제 해결 : object detector가 예측한 bounding box 중에서 정확한 bounding box를 선택하도록 하는 값
        self.conf_thres = conf_thres

        self.resize_fn = ResizePadding(input_size, input_size)
        self.transf_fn = transforms.ToTensor() # 데이터를 텐서 형식으로 변환

    def detect(self, image, need_resize=True, expand_bb=5):
        """Feed forward to the model.
        Args:
            image: (numpy array) Single RGB image to detect.,
            need_resize: (bool) Resize to input_size before feed and will return bboxs
                with scale to image original size.,
                피드하기 전에 input size로 크기를 조정하고 이미지 원본 크기로 크기를 조정하여 bbox 반환
            expand_bb: (int) Expand boundary of the boxs.
            boundary box 확장
        Returns:
            (torch.float32) Of each detected object contain a
                [top, left, bottom, right, bbox_score, class_score, class]
            return `None` if no detected.
        """
        image_size = (self.input_size, self.input_size) # 384 X 384
        if need_resize:                  # need_resize : False
            image_size = image.shape[:2] # (height, width)
            image = self.resize_fn(image) # image를 설정된 크기로 변경

        image = self.transf_fn(image)[None, ...] # image를 tensor 형식으로 변경
        scf = torch.min(self.input_size / torch.FloatTensor([image_size]), 1)[0] # tensor에 있는 모든 요소의 최소값

        detected = self.model(image.to(self.device))
        detected = non_max_suppression(detected, self.conf_thres, self.nms)[0] # conf_thres : 0.45, nms : 0.2 : 가장 정확한 bbox
        if detected is not None: # 기존 image 크기가 아니라 신경망의 입력 사이즈이기 때문에 bbox의 꼭지점 인자를 원래 차원으로 변환
            detected[:, [0, 2]] -= (self.input_size - scf * image_size[1]) / 2
            detected[:, [1, 3]] -= (self.input_size - scf * image_size[0]) / 2
            detected[:, 0:4] /= scf

            detected[:, 0:2] = np.maximum(0, detected[:, 0:2] - expand_bb)
            detected[:, 2:4] = np.minimum(image_size[::-1], detected[:, 2:4] + expand_bb)

        return detected


class ThreadDetection(object):
    def __init__(self,
                 dataloader,
                 model,
                 queue_size=256):
        self.model = model

        self.dataloader = dataloader
        self.stopped = False
        self.Q = Queue(maxsize=queue_size)

    def start(self):
        t = Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            images = self.dataloader.getitem()

            outputs = self.model.detect(images)

            if self.Q.full():
                time.sleep(2)
            self.Q.put((images, outputs))

    def getitem(self):
        return self.Q.get()

    def stop(self):
        self.stopped = True

    def __len__(self):
        return self.Q.qsize()







