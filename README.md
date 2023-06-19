# falling_detect_radar

RMPE(Alphapose): https://arxiv.org/abs/1612.00137<br>
ST-GCN: https://arxiv.org/abs/1709.04875


## YOLOv3 + Alphapose + ST-GCN + Radar
![image](https://github.com/KYUJEONG98/falling_detect_radar/assets/87844641/91e933bb-cc25-4155-9b52-a604ca360dc4)


자세 추정 모델의 자세 확률에, 머신 러닝을 통해 생성한 레이더 센서 낙상 판단 모델의 판단값을 
가중치로 적용한 낙상 판단 모델 연구



## Radar 낙상 판단 모델 
낙상과 낙상이 아닌 자세에 대해 레이더 데이터를 측정하여 
머신러닝 학습을 통해 생성한 실시간 레이더 센서 낙상 판단이 가능한 모델

해당 모델을 ST-GCN 포즈확률에 가중치를 적용해 낙상 판단 정확도를 상승시킴 





## Conlcusion
<img width="487" alt="정확도" src="https://github.com/KYUJEONG98/falling_detect_radar/assets/101076275/8cc1a500-19ca-48d6-97e3-86ec2fa3dcec">


자세 추정 모델과 레이더 센서 낙상 판단 모델을 결합하였을 때, 
낙상 판단 정확도 상승을 확인 
