class opt:
     nClasses = 33 # class 수 
     inputResH = 384 #입력 이미지 높이
     inputResW = 320 # 입력 이미지의 너비
     outputResH = 96 # 출력 heatmap의 높이
     outputResW = 80 # 출력 heatmap의 너비
     scale = 0.25 # 이미지 크기 조정 비율
     rotate = 30 # 이미지 회전 각도
     hmGauss = 1 # heatmap을 생성할 때 Gaussian 필터의 크기

