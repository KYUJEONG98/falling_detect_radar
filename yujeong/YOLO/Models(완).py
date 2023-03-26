import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .Utils import build_targets, to_cpu, parse_model_config


# 모듈 생성 함수로 nn.ModuleList형태로 모듈 리스트를 만들고 반환
def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    # hyperparams는 첫 번째 딕셔너리로 입력 이미지 크기,채널 수,학습률,배치 크기 등 구성 
    hyperparams = module_defs.pop(0) 
    output_filters = [int(hyperparams["channels"])]  # [3]
    module_list = nn.ModuleList()
    
    # module_defs에서 탐색하고 해당 레이어를 module에 추가한다
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        # 컨볼루션 처리
        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn: #batch_normalize 필드가 1일 경우 배치 정규화 추가
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky": #nn.LeakyReLU 추가
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))


        # MaxPool 처리
        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            
            #zeroPad는 MaxPooling이 차원을 줄이기에 피쳐맵 크기를 늘리는 역할
            if kernel_size == 2 and stride == 1: 
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, 
                                   stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)


        # Upsample 처리
        elif module_def["type"] == "upsample":
                
            #nearest는 가장 가까운 픽셀 선택
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)


        # route 레이어
        # 이전 레이어의 출력을 가져와서 새로운 텐서를 만드는 레이어
        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            
            # outfilter는 모든 레이어의 출력 필터 수를 저장하는 리스트로
            # layer에서 가져온 인덱스에 해당하는 레이어의 출력 필터 수를 모두 더함
            filters = sum([output_filters[1:][i] for i in layers])
            # filters는 route 모듈의 채널 수 <- 다음 레이어의 입력 채널 수로 사용
            modules.add_module(f"route_{module_i}", EmptyLayer())

        # shortcut은 이전 레이어의 출력값을 현재 레어어의 출력값에 더함
        # 따라서 이전 레이어의 출력 필터 수 필요
        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            # shortcut 레이어는 아무 연산을 수행하지 않기에 EmptyLayer 사용
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())

        # YOLO 레이어
        elif module_def["type"] == "yolo":
            
            # mask는 객체 검출에서 사용되는 앵커 박스의 크기와 비율을 지정
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            img_size = int(hyperparams["height"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


# 특성 맵의 크기를 늘리는 등의 용도
class Upsample(nn.Module):
    """ nn.Upsample is deprecated """
    # scale_factor는 입력 데이터를 얼마나 확대할지를 결정하는 값
    # nearest는 최근접 이웃 보간 방식으로 UpSampling 수행
    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
    
    def forward(self, x):
        # interpolate를 사용해 업샘플링 수행
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        """
        interpolate
        이미지를 확대하거나 축소할 때 사용되며, 이미지를 이루는 픽셀 값들을
        새로운 이미지의 크기에 맞게 보간하는 과정을 의미 
        
        입력 이미지에서 새로운 이미지로 보간되는 각 픽셀은 가장 가까운 픽셀 값으로 결정
        즉, 각 픽셀은 입력 이미지에서 가장 가까운 픽셀 값으로 결정
        계산이 간단하고 빠르지만, 업샘플링 과정에서 이미지의 선명도가 떨어짐
        """
        return x

# route와 shortcut 레이어를 나타내는 레이어
# 입력값을 
class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""
    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module): #def creat module에 참조
    """Detection layer
    입력 이미지를 그리드로 분할하고 분할된 각 cell에 대해 object detection 결과 예측

    객체가 존재할 가능성이 높은 지역(region proposals)
    
    1. bounding box
    (bouding box regression은 입력 이미지 내에서 특정 객체의 bounding box의 위치와 크기의 좌표 
    예측, 이 좌표를 사용하여 실체 객체 위치와 가능한 한 정확하게 일치하도록 bouding box 조정)

    2. class probability
    3. objectness score
    """
    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors #앵커 박스의 width와 height
        self.num_anchors = len(anchors) # 앵커박스 수 
        self.num_classes = num_classes # 분류할 객체의 수
        self.ignore_thres = 0.5 # IOU 스레쉬홀드 값
        
        # MSE는 bounding box regression 손실을 계산
        self.mse_loss = nn.MSELoss() # 예측값과 실제값의 차이를 측정 LOSS 함수
        
        # BCE는 objectness classification 손실을 계산
        self.bce_loss = nn.BCELoss() # 예측값과 실제값의 차이를 측정 LOSS 함수
        
        self.obj_scale = 1 # objectness score의 가중치
        self.noobj_scale = 100 
        self.metrics = {} # 학습 도중 사용할 metrics
        self.img_dim = img_dim # 입력 이미지 크기
        self.grid_size = 0  # grid size

    # grid size와 이미지 크기를 이용해 bouding box들이 매핑될 각각의 grid cell의 위치 계산
    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        
        # 앵커박스의 width와 height를 현재 feature map에 맞게 크기 조정
        # a_w,a_h는 한 cell의 크기
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        
        # 모든 행을 선택하고 열은 0부터 1번->(num_anchors,1) 형태 텐서
        # 배치 크기, 앵커 박스 수, 그리드 셀 높이 수, 너비 수
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):
        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0)# 배치 크기-> x에 포함된 이미지의 개수
        grid_size = x.size(2) # 이미지의 grid 크기

        # outlayer의 출력값을 변환하는 과정 수행
        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        ) # x를 5차원으로 변경->첫 번째 축:num_samples, 2번째:num_anchors, 3번쨰:grid_size ..

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])# bounding box의 중심 좌표 x값
        y = torch.sigmoid(prediction[..., 1])# bounding box의 중심 좌표 y값
        w = prediction[..., 2]# bounding box의 너비
        h = prediction[..., 3]# bounding box의 높이
        pred_conf = torch.sigmoid(prediction[..., 4])#bounding box가 있을 확률
        pred_cls = torch.sigmoid(prediction[..., 5:])#bounging box안에 있는 객체의 종류에 대한 확률
        # 클래스의 개수에 따라 출력값의 차원이 달라짐

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        #bounding box의 좌표와 크기 계산
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        #self.grid_x, self.grid_y는 이미 계산된 grid cell의 좌상단 좌표
        pred_boxes[..., 0] = x.data + self.grid_x# x를 더해 bounding box의 중심점 좌표 구함
        pred_boxes[..., 1] = y.data + self.grid_y# y를 더해 bounding box의 중심점 좌표 구함
        
        # w,h는 실제 너비와 높이가 아닌 anchor box의 너비와 높이에 대한 상대적인 비율임
        # 지수함수를 취하여 실제 크기로 변환
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        # YOLO의 최종 출력값
        output = torch.cat(
            (
                #각 앵커박스에 대한 바운딩 박스 예측값을 2D로 변환
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                
                # bounding box에 대한 cofidence score를 2D텐서로 변환
                pred_conf.view(num_samples, -1, 1), # 1은 SCORE가 하나의 값이기에
                
                #각 클래스에 대한 예측 확률을 2D로 변환, num_classes는 클래스 수
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1, #마지막 차원으로 예측값의 모든 차원을 유지한 채 하나의 텐서로 합침
        )

        # YOLOv3의 학습 수행 
        if targets is None:
            return output, 0
        else: # build_targets를 사용해 각 예측 박스에 대한 ground truth를 생성
              # 이를 기반으로 각 예측값과 ground truth간의 손실을 계산해 총 손실을 구함
              # 이후 역전파하여 가중치 업데이트
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            # obj_mask와 noobj_mask는 각 위치의 바운딩 박스와 연관된 ground truth 박스의 존재 여부 나타냄
            # obj_mask -> 존재하는 ground truth 박스에 대한 위치 예측 학습
            # noobj_mask -> 존재하지 않는 ground truth 박스에 대한 위치 예측 학습
            loss_x = self.mse_loss(x[obj_mask.bool()], tx[obj_mask.bool()])
            loss_y = self.mse_loss(y[obj_mask.bool()], ty[obj_mask.bool()])
            loss_w = self.mse_loss(w[obj_mask.bool()], tw[obj_mask.bool()])
            loss_h = self.mse_loss(h[obj_mask.bool()], th[obj_mask.bool()])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask.bool()], tconf[obj_mask.bool()])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask.bool()], tconf[noobj_mask.bool()])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask.bool()], tcls[obj_mask.bool()])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
            # loss가 최소화하는 방향으로 모델 학습
            
            
            # Metrics
            #예측한 클래스 중 정답 클래스의 비울 계산
            cls_acc = 100 * class_mask[obj_mask.bool()].mean()
            
            # 정답 박스에 대한 예측 확률의 평균값 
            conf_obj = pred_conf[obj_mask.bool()].mean()
            
            # 배경에 대한 예측 확률의 평균값
            conf_noobj = pred_conf[noobj_mask.bool()].mean()
            
            # 예측 확률이 0.5이상인 예측값에 대해 1 아니면 0
            conf50 = (pred_conf > 0.5).float()
            
            # loU가 0.5이상이면 1 아니면 0
            iou50 = (iou_scores > 0.5).float()
            
            # iou가 0.75이상이면 1 아니면 0
            iou75 = (iou_scores > 0.75).float()
            
            # 예측값 중에서 정답 박스에 대한 예측값은 1, 아니면 0
            detected_mask = conf50 * class_mask * tconf
            
            #예측값 중 ioU가 0.5이상, 정답 박스에 대한 예측값은 1 아니면 0인 예측확률의 평균값
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            
            # 중간결과 보여줌
            self.metrics = {
                "loss": to_cpu(total_loss).item(), # 학습 손실 값
                "x": to_cpu(loss_x).item(),# 박스 중심 x좌표 예측 손실 값
                "y": to_cpu(loss_y).item(),# 박스 중심 y좌표 예측 손실 값
                "w": to_cpu(loss_w).item(),# 박스 너비의 예측 손실 값
                "h": to_cpu(loss_h).item(),# 박스 높이의 예측 손실 값
                "conf": to_cpu(loss_conf).item(),# confidence 예측 손실 값
                "cls": to_cpu(loss_cls).item(),# 클래스 예측 손실 값
                "cls_acc": to_cpu(cls_acc).item(),# 클래스 정확도 값
                "recall50": to_cpu(recall50).item(),# ioU가 0.5이상인 박스에 대한 값
                "recall75": to_cpu(recall75).item(),# ioU가 0.75이상인 박스에 대한 값
                "precision": to_cpu(precision).item(),# precision값
                "conf_obj": to_cpu(conf_obj).item(),# object cofidence 값
                "conf_noobj": to_cpu(conf_noobj).item(),# non-object confidence 값
                "grid_size": grid_size,                # 그리드 크기 값
            }
            # precision값이 높을수록 모델이 예측한 바운딩 박스가 실제로 객체를 잘 포함하고 있음
            # recall값이 높을수록 모델이 실제 객체를 잘 감지
            return output, total_loss


class Darknet(nn.Module):
    """YOLOv3 object detection model"""
    
    #config파일과 이미지 입력받음
    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        
        # YOLO 모델은 self.module_list로 구성됨
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        
        #metrics 속성은 모델이 학습하면서 사용하는 지표들 의미
        #metrics의 포함하는 레이어만 따로 추출하여 yolo layer에 추가
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
        self.img_size = img_size
        self.seen = 0 # 현재까지 학습된 이미지의 수, 새로운 학습을 시작할 때 0으로 초기화
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    # 입력 이미지 x와 정답 라벨 targets로 YOLO 출력값 반환
    def forward(self, x, targets=None):
        img_dim = x.shape[2] #입력 이미지 크기
        loss = 0
        #layer_output은 layer 출력값 저장, yolo_outputs은 scale 출력값 저장
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
                
            # 이전 레이어의 출력값을 연결(concatenate)하여 현재 레이어의 입력값으로 사용
            # 이전 레이어의 출력값들을 가져와서 concatenate하는 이유는, 
            # 모델의 특징 맵(feature map)을 다양한 스케일(scale)에서 결합하면서 더욱 복잡한 특징들을 추출하기 위해서
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            
            # residual 구현
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                # layer_outputs[-1]은 이전 레이어의 출력값
                # layer_outputs[layer_i]은 from으로 지정된 인덱스의 레이어 출력값
                x = layer_outputs[-1] + layer_outputs[layer_i]
                
            # YOLO 레이어
            # feature map으로 bounding box의 좌표와 클래스 확률 예측
            # 예측 값으로 실제 객체를 인식하고 위치 파악
            elif module_def["type"] == "yolo":
                # x는 yolo layer 이전의 최종 feature map
                # targets는 ground truth인 라벨
                # img_dim은 입력 이미지의 크기
                x, layer_loss = module[0](x, targets, img_dim)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)
            
        # 모든 YOLO 레이어의 출력을 연결
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    #Darknet 모델에서 사전 학습된 weights 파일을 parsing하고 해당 가중치를 모델의 각 레이어에 로드
    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""
        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        # 뒷부분에 있는 레이어 중 75번째까지만 로드하도록 cutoff를 설정
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75


        # Darknet으로 학습된 weights를 PyTorch 모델로 변환에서 사용용
        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    # darknet 형식의 가중치를 저장하는 함수
    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
            ^- cutoff이 -1이면 모든 레이어를 저장
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()

    # 미리 학습된 PyTorch 모델의 가중치(weights)를 커스텀 클래스의 가중치로 로드하는 함수
    def load_pretrain_to_custom_class(self, weights_pth_path):
        state = torch.load(weights_pth_path)

        own_state = self.state_dict() # 커스텀 클래스의 가중치를 불러옴
        for name, param in state.items():
            if name not in own_state:
                print(f'Model does not have this param: {name}!')
                continue

            if param.shape != own_state[name].shape:
                print(f'Do not load this param: {name} cause it shape not equal! : '
                      f'{param.shape} into {own_state[name].shape}')
                continue

            own_state[name].copy_(param)
