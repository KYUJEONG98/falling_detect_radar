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
        input_size: (int)입력 이미지의 크기, 32로 나누어 떨어져야 함. Default: 416,
        config_file: (str) YOLO 모델 구조 설정 파일의 경로,
        weight_file: (str) 사전 학습된 가중치 파일의 경로,
        nms: (float) Non-Maximum Suppression(비최대 억제) 중첩 임계값,
        conf_thres: (float) 예측된 bbox의 최소 확률 임계값,
        device: (str) Device to load the model on 'cpu' or 'cuda'.
    """
    def __init__(self,
                 input_size=416,
                 config_file='Models/yolo-tiny-onecls/yolov3-tiny-onecls.cfg',
                 weight_file='Models/yolo-tiny-onecls/best-model.pth',
                 nms=0.2,
                 conf_thres=0.45,
                 device='cuda'):
        self.input_size = input_size
        self.model = Darknet(config_file).to(device)
        self.model.load_state_dict(torch.load(weight_file))
        self.model.eval()
        self.device = device

        self.nms = nms
        self.conf_thres = conf_thres

        self.resize_fn = ResizePadding(input_size, input_size)
        self.transf_fn = transforms.ToTensor()

    # RGB 이미지를 입력받아, 객체를 검출
    def detect(self, image, need_resize=True, expand_bb=5):
        """Feed forward to the model.
        Args:
            image: (numpy array) Single RGB image to detect.,
            need_resize: (bool) need_resize가 True이면 input size로 크기 조정,
            expand_bb: (int) bbox를 이미지 경계를 벗어나게 확장시키는 값
                        이 값으로 bbox가 실체 물체의 경계까지 포함되도록 함
                        -> 검출된 객체 주위에 여유 공간이 있도록 확장하여 이미지 경계를 벗어나는
                            부분의 객체를 놓치지 않도록 함
        Returns:
            (torch.float32) Of each detected object contain a
                [top, left, bottom, right, bbox_score, class_score, class]
            return `None` if no detected.
        """
        image_size = (self.input_size, self.input_size)
        if need_resize:
            image_size = image.shape[:2]
            image = self.resize_fn(image)

        image = self.transf_fn(image)[None, ...]
        # scale factor로 입력 이미지 크기와 모델의 입력 크기 사이의 비울
        scf = torch.min(self.input_size / torch.FloatTensor([image_size]), 1)[0]

        # 모델을 통해 bounding box detection 수행
        detected = self.model(image.to(self.device))# 이미지를 모델에 통과시켜 detected에 저장
        
        # nms를 통해 중복 bbox 제거와 conf_thes 이하의 bbox 제거
        detected = non_max_suppression(detected, self.conf_thres, self.nms)[0]
        if detected is not None:
            # bbox 좌상단
            detected[:, [0, 2]] -= (self.input_size - scf * image_size[1]) / 2
            # bbox 우하단
            detected[:, [1, 3]] -= (self.input_size - scf * image_size[0]) / 2
            
            # bbox의 좌표값 scf로 조정
            detected[:, 0:4] /= scf

            detected[:, 0:2] = np.maximum(0, detected[:, 0:2] - expand_bb)
            detected[:, 2:4] = np.minimum(image_size[::-1], detected[:, 2:4] + expand_bb)

        return detected # bbox(top,left,bottom,right,bbox_score,class_score,class) 반환

# object detection을 수행하는 thread 정의 클래스
class ThreadDetection(object):
    def __init__(self,
                 dataloader,
                 model,
                 queue_size=256): # 스레드에서 생성하는 큐의 max size
        self.model = model

        self.dataloader = dataloader
        self.stopped = False
        self.Q = Queue(maxsize=queue_size)

    def start(self): # 스레드 시작
        t = Thread(target=self.update, args=(), daemon=True).start()
        return self

    # dataloader에서 이미지를 가져와 모델을 사용해 객체 검출하여 큐에 추가
    # 큐가 차면 2초간 sleep
    def update(self):
        while True:
            if self.stopped:
                return

            images = self.dataloader.getitem()

            outputs = self.model.detect(images) #모델 검출

            if self.Q.full():
                time.sleep(2)
            self.Q.put((images, outputs))

    def getitem(self):
        return self.Q.get()

    def stop(self):
        self.stopped = True

    def __len__(self):
        return self.Q.qsize()







