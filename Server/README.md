# Radar

레이더는 파워로직스사의 제품으로 "TSF-P100" 모델을 사용하였고, 물체를 감지하고 움직임이 있을 때 데이터를 전송한다.<br>
레이더 데이터는 아래와 같이 구성된다.
* posX
  + 현재 Target의 x 좌표점으로 10cm단위
* posY
  + 현재 Target의 y 좌표점으로 10cm단위
* posZ
  + 현재 Target의 z 좌표점으로 10cm단위이지만 2D에서의 값은 0이다.
* bpm
  + 사용안함
* hbr
  + 사용안함
* engergy
  + 표적의 운동 에너지량	
  + Target에 대한 Energy값을 표현한 것으로 동적움직임의 크기를 나타낸다. 
  + 0일수 있음

<br>
<br>

## 레이더 데이터를 받기 위한 서버
#### Mobius IoT/M2M Server
#### &Cube: Thyme, Lavender -> 단말
#### &Cube: Rosemart -> 게이트웨이
#### TAS(Thing Adaption Software)
