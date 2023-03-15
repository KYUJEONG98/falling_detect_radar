from torch import nn


     
class SELayer(nn.Module):
     """_summary_
          SE 블록은 오버피팅, 기울기 소실 문제를 해결하기 위한
          Squeeze, Excitation 연산으로 구성
          
          1. Squeeze 연산
          - input의 공간 차원을 제거하고 채널 축을 기반으로 통계량 계산
          - (ex. 평균, 최댓값)
          - (채널 축은 RGB의 3과 같은)
          - 채널마다 정보를 압축하고 특성을 나타내는 벡터를 생성
          - 이후 Excitation 연산에서 사용

          Input 데이터 x = (B,C,H,W)=(배치크기, 채널 수, 높이, 너비)
          =>(B, C, 1, 1)로 크기 변경<- nn.AdaptiveAvgPool1d(1) 사용
          이는 입력 데이터의 채널마다 평균 값을 계산할 수 있음
          
          
          ***Squeeze 연산 결과는 (B,C,1,1)이 됨***
          
          
          2. Excitation 연산
          Squeeze 연산에서 생성된 채널마다의 특성을 나타내는 벡터를
          사용하여 input의 각 채널을 재설정
          
          - Squeeze 결과인 (B,C,1,1)을 nn.Linear 및 활성화 함수 사용
          - FC Layer를 사용해 각 채널의 중요도 계산
          - 또한 채널의 중요도를 학습하고 강조 or 무시하는 가중치 생성
          - 가중치는 (B,C,1,1)인 벡터로 출력
          - input은 가중치와 원소별 곱셈을 계산함
          
          
          ***output은 입력 데이터의 중요한 채널을 강조하거나 무시***
     """
     def __init__(self,channel,reduction=1):
          super(SELayer,self).__init__()
          self.avg_pool = nn.AdaptiveAvgPool1d(1)
          self.fc = nn.Sequential(
               nn.Linear(channel, channel // reduction),
               nn.ReLU(inplace=True),
               nn.Linear(channel // reduction,channel),
               nn.Sigmoid()
          )
     
     def foward(self, x):
          b, c, _, _ = x.size()
          y=self.avg_pool(x).view(b,c) # Squeeze 연산
          y=self.fc(y).view(b,c,1,1)   # Excitation 연산
          return x*y     # input과 가중치의 곱을 return