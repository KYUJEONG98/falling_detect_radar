# falling_detect_radar

RMPE(Alphapose): https://arxiv.org/abs/1612.00137<br>
ST-GCN: https://arxiv.org/abs/1709.04875


## 기존의 문제
유치장을 포함한 실내공간에서 낙상 사고가 일어나고 있으나, 인력 및 기술의 문제로 실시간 파악이 어려운 상황임

이에 대해 자세 추정 모델을 통한 낙상 판단에 방법이 제기되었지만, 이는 정면에서 촬영한 이미지를 대상으로 하는 것으로
실제 천장 가장자리에서 촬영하는 CCTV 영상에서 정확도가 높지 않음

따라서 레이더 센서 기기를 통해 이를 보정하는 방법을 찾고자함 


## YOLOv3 + Alphapose + ST-GCN + Radar
![image](https://github.com/KYUJEONG98/falling_detect_radar/assets/87844641/91e933bb-cc25-4155-9b52-a604ca360dc4)


자세 추정 모델의 자세 확률에, 머신 러닝을 통해 생성한 레이더 센서 낙상 판단 모델의 판단값을 
가중치로 적용한 낙상 판단 모델 연구



## Radar 낙상 판단 모델 
낙상과 낙상이 아닌 자세에 대해 레이더 데이터를 측정해 약 4600개의 데이터 생성

이를 머신러닝 학습하여 실시간 레이더 센서 낙상 판단이 가능한 모델 제작

해당 모델의 결과를 ST-GCN 자세 확률에 가중치곱을 적용해 최종 낙상 판단 정확도를 상승시킴 





## Conlcusion
<img width="487" alt="정확도" src="https://github.com/KYUJEONG98/falling_detect_radar/assets/101076275/8cc1a500-19ca-48d6-97e3-86ec2fa3dcec">


자세 추정 모델과 레이더 센서 낙상 판단 모델을 결합하였을 때, 
낙상 판단 정확도 상승을 확인 
