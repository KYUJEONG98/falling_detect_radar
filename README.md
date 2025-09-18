# falling_detect_radar

RMPE(Alphapose): https://arxiv.org/abs/1612.00137<br>
ST-GCN: https://arxiv.org/abs/1709.04875


## 기존의 문제
유치장을 포함한 실내공간에서 낙상 사고가 일어나고 있으나, 인력 및 기술의 문제로 실시간 파악이 어려운 상황임

이에 대해 자세 추정 모델을 통한 낙상 판단에 방법이 제기되었지만, 이는 정면에서 촬영한 이미지를 대상으로 하는 것으로
실제 천장 가장자리에서 촬영하는 CCTV 영상에서 정확도가 높지 않음

따라서 레이더 센서 기기를 통해 이를 보완하고자함 


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

[2023 대한전자공학회 하계학술대회 논문 링크](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11522430&nodeId=NODE11522430&mobileYN=N&medaTypeCode=185005&isPDFSizeAllowed=true&locale=ko&foreignIpYn=N&articleTitle=CCTV+%EC%99%80+%EB%A0%88%EC%9D%B4%EB%8D%94+%EC%84%BC%EC%84%9C%EB%A5%BC+%ED%99%9C%EC%9A%A9%ED%95%9C+%EC%8B%A4%EB%82%B4%EC%97%90%EC%84%9C%EC%9D%98+%EB%82%99%EC%83%81+%EC%82%AC%EA%B3%A0+%EA%B0%90%EC%A7%80+%EB%B0%A9%EB%B2%95&articleTitleEn=A+Method+of+Fall+Accidents+Detection+Using+CCTV+and+Radar+Sensor+in+the+Room&voisId=VOIS00727004&voisName=2023%EB%85%84%EB%8F%84+%EB%8C%80%ED%95%9C%EC%A0%84%EC%9E%90%EA%B3%B5%ED%95%99%ED%9A%8C+%ED%95%98%EA%B3%84%ED%95%99%EC%88%A0%EB%8C%80%ED%9A%8C+%EB%85%BC%EB%AC%B8%EC%A7%91&voisCnt=856&language=ko_KR&hasTopBanner=true) 
