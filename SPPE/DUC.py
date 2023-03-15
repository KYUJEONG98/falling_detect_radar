import torch.nn as nn
import torch.nn.functional as F


class DUC(nn.Module): 
     """
          INPUT: inplanes, planes, upscale_factor
          OUTPUT: (planes // 4) * ht * wd
          
          Dense Upsampling Convolution(DUC)은 입력된 이미지를
          밀집하게 Upsampling하는 연산<-축소된 feature Map을 원본 크기로 돌리기 위해
     
          PixelShuffle을 통해 upsampling됨
            <- 크기를 늘리고 feature map의 정보를 밀집하게 유지함
     """
     def __init__(self, inplanes,planes,upscale_factor=2):
          super(DUC,self).__init__()
          self.conv = nn.Conv2d(inplanes,planes,kernel_size=3
                                ,padding=1,bias=False)
          self.bn = nn.BatchNorm2d(planes)
          self.relu =nn.ReLU()
          self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
          # nn.PixelShuffle은 이미지를 일정한 크기의 블록으로 나눈 후,
          # 이를 합치는 과정에서 정보를 보존하면서 이미지의 해상도를 높임
          # 블록 단위로 이미지를 나누는 대신, 블록을 연결하는 것으로 이미지를 재배치 
     def foward(self,x):
          x = self.conv(x)
          x = self.bn(X)
          x = self.relu(x)
          x = self.pixel_shuffle(x)
          return x