import os
import torch
import numpy as np

from Actionsrecognition.Models import TwoStreamSpatialTemporalGraph
from pose_utils import normalize_points_with_size, scale_pose


class TSSTG(object):
    """Two-Stream Spatial Temporal Graph Model Loader.
    Args:
        weight_file: (str) Path to trained weights file. 
        훈련된 가중치 파일의 경로
        device: (str) Device to load the model on 'cpu' or 'cuda'.
        'cpu' 또는 'cuda'에서 모델을 로드
    """
    def __init__(self,
                 weight_file='./Models/TSSTG/tsstg-model.pth',
                 device='cuda'):
        self.graph_args = {'strategy': 'spatial'}
        self.class_names = ['Standing', 'Walking', 'Sitting', 'Lying Down',
                            'Stand up', 'Sit down', 'Fall Down'] #모델의 class
        self.num_class = len(self.class_names)
        self.device = device

        self.model = TwoStreamSpatialTemporalGraph(self.graph_args, self.num_class).to(self.device) #모델
        self.model.load_state_dict(torch.load(weight_file)) #모델 불러오기
        self.model.eval() #모델 평가모드로 바꿈

    def predict(self, pts, image_size):
        """Predict actions from single person skeleton points and score in time sequence.
        Args:
            pts: (numpy array) points and score in shape `(t, v, c)` where
                t : inputs sequence (time steps).입력 시퀀스(시간),
                v : number of graph node (body parts).그래프 노드(본체 부분)의 수,
                c : channel (x, y, score).채널(x, y, 점수),
            image_size: (tuple of int) width, height of image frame.
            이미지 프레임의 사이즈(넓이, 높이).
            
        Returns:
            (numpy array) Probability of each class actions.
            추정되는 행동 class의 확률
        """
        pts[:, :, :2] = normalize_points_with_size(pts[:, :, :2], image_size[0], image_size[1]) #이미지 정규화
        pts[:, :, :2] = scale_pose(pts[:, :, :2]) #이미지 정규화
        pts = np.concatenate((pts, np.expand_dims((pts[:, 1, :] + pts[:, 2, :]) / 2, 1)), axis=1) #배열 합치기

        pts = torch.tensor(pts, dtype=torch.float32)
        pts = pts.permute(2, 0, 1)[None, :] #차원 순서 바꾸기

        mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :] #multiple object tracking 다중 객체 트래킹
        mot = mot.to(self.device) #CUDA에 최적화
        pts = pts.to(self.device) #CUDA에 최적화

        out = self.model((pts, mot))

        return out.detach().cpu().numpy() 
        #detach():tensor 반환, cpu():tensor를 cpu메모리로 복사, numpy():cpu에 올라와 있는 tensor를 numpy로 변환
