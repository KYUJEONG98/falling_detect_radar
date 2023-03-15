import torch.nn as nn
from .SE_module import SELayer
import torch.nn.functional as F

class Bottleneck(nn.Module):
     """_summary_
     Convolution parameter = kernel_size(가로) x kernel_size(세로) x input channel x output channel
     BottleNeck의 핵심은 1x1 conv(= Pointwise conv라고도 함)
     -> 연산량이 작기에 feature map을 줄이거나 키울 때 사용
     
     ResNet에서 사용되는 핵심구조
     
     """
     expansion = 4
     # inplanes는 in_channel, planes=out_channel
     def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=False):
          super(Bottleneck, self).__init__()
          self.conv1 = nn.Conv2d(inplanes,planes,kernel_size=1,bias=False)
          # 1x1 conv를 통해 채널 수를 줄임
          self.bn1= nn.BatchNorm2d(planes)
          self.conv2 = nn.Conv2d(planes,planes, kernel_size=3, stride=stride,
                                 padding=1, bias=False)
          # 3x3 conv를 통해 feature map의 크기를 줄이고 입력 채널 수 늘림
          self.bn2 = (planes)
          self.conv3 = nn.Conv2d(planes,planes*4, kernel_size=1, bias=False)
          # 1x1 conv를 통해 feature map의 크기를 다시 줄이고 출력 채널 수를 증가
          # 이 과정에서 feature map의 크기가 작아져 높은 정확도와 빠른 속도
          self.bn3 = nn.BatchNorm2d(planes*4)
          if reduction:
               self.se=SELayer(planes*4)
               
          self.reduc= reduction
          self.downsample = downsample
          self.stride = stride
     
     def foward(self,x):
          residual = x
          
          out = F.relu(self.bn1(self.conv1(x)), inplace=True)
          out = F.relu(self.bn2(self.conv2(out)), inplace=True)
          
          out = self.conv3(out)
          out = self.bn3(out)
          if self.reduc:
               out = self.se(out)
          
          if self.downsample is not None:
               residual = self.downsample(x)
               
          out += residual # F(x) + x
          out = F.relu(out)
          
          return out

class SEResnet(nn.Module):
     def __init__(self, architecture):
          super(SEResnet, self).__init__()
          
          #resnet50 이나 resnet101 둘 중 하나 사용
          assert architecture in ["resnet50", "resnet101"]
          # 첫 번째 convolutional layer에서 출력되는 채널 수
          self.inplanes = 64
          
           #각각의 레이어에서 사용할 bottleneck의 수->layer1,2,3,4
           #resnet50은 3,4,6,3---- resnet101은 3,4,23,3
          self.layers = [3, 4, {"resnet50": 6, "resnet101": 23}[architecture], 3]
          self.block=Bottleneck
          
          self.conv1 = nn.Conv2d(3,64,kernel_size=7,
                                 stride=2, padding=3, bias = False)
          self.bn1 = nn.BatchNorm2d(64,eps=1e-5,momentum=0.01,padding=1)
          self.relu = nn.ReLU(inplace=True)
          self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
          
          #layer에서 
          self.layer1 = self.make_layer(self.block, 64, self.layers[0])
          self.layer2 = self.make_layer(
          self.block, 128, self.layers[1], stride=2)
          self.layer3 = self.make_layer(
          self.block, 256, self.layers[2], stride=2)
          self.layer4 = self.make_layer(
          self.block, 512, self.layers[3], stride=2)
          
     def forward(self, x):
          x = self.maxpool(self.relu(self.bn1(self.conv1(x))))  # 64 * h/4 * w/4
          x = self.layer1(x)  # 256 * h/4 * w/4
          x = self.layer2(x)  # 512 * h/8 * w/8
          x = self.layer3(x)  # 1024 * h/16 * w/16
          x = self.layer4(x)  # 2048 * h/32 * w/32
          return x
     
     def stages(self):
          return [self.layer1, self.layer2, self.layer3, self.layer4]

     #block은 bottleneck의 수
     def make_layer(self, block, planes, blocks, stride=1):
          """각 layer를 생성하는 역할 
          block 인자는 ResNet에서 사용할 블록 클래스 받아옴
          planes는 layer에서 사용할 필터의 개수
          blocks는 layer 내에서 사용할 블록의 개수
          stride는 첫 번째 블록에서 사용할 stride 값

          그 후, 첫 번째 블록을 생성합니다. 
          downsample이 존재하면, 첫 번째 블록에서는 reduction=True로 설정하여, 
          입력 feature map의 크기가 줄어든 경우 shortcut connection에서 
          channel 수를 맞춰주기 위한 1x1 Convolution 연산을 추가합니다. 
          이전 layer에서의 출력 feature map과 다음 layer에서 사용할 feature map의 channel 수가 다르지 않은 경우, 
          

          """
          downsample = None #downsample 변수 초기화
          
          # 만약 stride가 1이 아니거나, 이전 layer에서의 출력 feature map과 
          # 다음 layer에서 사용할 feature map의 channel 수가 다르다면, downsample을 구성
          if stride != 1 or self.inplanes != planes * block.expansion:
               downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
          # downsample은 feature map의 크기를 조정해주기 위해 사용
          # (downsample은 Convolution과 BatchNormalization 층으로 구성)
          layers = []
          
          if downsample is not None:
            layers.append(block(self.inplanes, planes, stride, downsample, reduction=True))
          else:
               layers.append(block(self.inplanes, planes, stride, downsample))
               
          self.inplanes = planes * block.expansion
          for i in range(1, blocks): # 블록 추가
               layers.append(block(self.inplanes, planes))

          return nn.Sequential(*layers) # 생성된 layer를 nn.Sequential 클래스로 묶어 반환