import numpy as np
import cv2
import torch
import scipy.misc
from torchvision import transforms
import torch.nn.functional as F
from scipy.ndimage import maximum_filter
from PIL import Image
from copy import deepcopy
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt

"""
     1.to_torch(ndarray): ndarray를 tensor로 변환
     2.to_numpy(tensor) : tensor를 Numpy의 ndarray로 변환
     3.torch_to_im(img) : tensor인 img를 numpy의 ndarray로 변환
     4.im_to_torch(img) :이미지(ndarray)을 PyTorch tensor로 변환
     5.load_image(img_path):이미지를 RGB로 읽고 numpy에서 torch로 변환
     6.cropBox(img,ul,br,resH,resW): 이미지를 일정한 크기로 자르고 작으면 
                                     padding을 통해 크기를 키움
"""

"""
NumPy의 ndarray와 PyTorch의 Tensor는 모두 
다차원 배열을 처리하는 데 사용되는 데이터 타입

1. 계산 그래프
 - NumPy ndarray는 계산 그래프를 지원하지 않습니다.
 - PyTorch Tensor는 계산 그래프를 지원합니다.
2. 연산
 - NumPy ndarray는 CPU 상에서 연산
 - PyTorch Tensor는 CPU or GPU 상에서 연산
3. 메모리 관리
 - NumPy ndarray는 메모리 별도 관리
 - PyTorch Tensor는 메모리 자동 관리
4. 데이터 타입
 - NumPy ndarray는 기본적으로 float64와 int64 등의 데이터 타입 사용
 - PyTorch Tensor는 float32와 int64 등의 데이터 타입 사용
 
따라서, PyTorch는 계산 그래프를 지원하여 복잡한 모델을 쉽게 구현. 
GPU로 빠른 연산을 지원, 메모리 관리도 자동으로 처리로 직접 메모리 관리x

"""
# 1. ndarray를 pytorch의 tensor로 변환해주는 함수
def to_torch(ndarray):      
     if type(ndarray).__module__ == 'numpy':
     # from_numpy는 ndarray를 tensor로 변환하는 함수
          return torch.from_numpy(ndarray)
     elif not torch.is_tensor(ndarray):
          raise ValueError("Cannot convert {} to torch tensor"
                           .format(type(ndarray)))
     return ndarray


# 2. tensor를 Numpy의 ndarray로 변환해주는 함수
def to_numpy(tensor):
     if torch.is_tensor(tensor):
          return tensor.cpu().numpy()
     elif type(tensor).__module__ != 'numpy':
          raise ValueError("Cannot convert {} to torch tensor"
                           .format(type(tensor)))
     return tensor
     
     
# 3. tensor인 img를 numpy의 ndarray로 변환하는 함수    
def torch_to_im(img):
     img = to_numpy(img)
     
     # (C,H,W)에서 (H,W,C)로 변경
     img = np.transpose(img, (1,2,0))
     # 첫 번째 축을 배열의 두번 째
     # 두 번쨰 축을 배열의 세번 쨰
     # 세 번쨰 축을 배열의 첫번 째로 변경하는 함수
     return img


# 4. 이미지 파일(ndarray)을 PyTorch tensor로 변환하는 함수
def im_to_torch(img):
     img = np.array(img) #이미지를 NUMPY로 변환
     # (H,W,C)를 (C,H,W)로 변경
     img = np.transpose(img, (2,0,1))
     img = to_torch(img).float() # NUMPY를 TENSOR로 변환
     
     #정규화 과정
     if img.max() > 1:
          img /= 255 #이미지 픽셀 값은 0~255
     return img


# 5.이미지를 RGB로 읽고 numpy에서 torch로 변환
def load_image(img_path):
     #scipy.misc.imread는 이미지 파일을 읽어 numpy배열로 읽음
     #이미지 각각의 픽셀의 RGB 값이 NumPy 배열의 값으로 대응
     return im_to_torch(scipy.misc.imread(img_path, mode='RGB'))


"""
     ***bounding box를 만드는 두 점의 좌표***
     - ul은 좌측 상단 꼭지점의 좌표
     - bl은 우측 하단 꼭지점의 좌표
"""     
# 원중심 좌표 pt, 원크기 sigma
def drawCircle(img,pt,sigma):
     # pt는 (x,y)처럼 주어질 듯
     img = to_numpy(img)
     tmpSize = 3*sigma #원의 크기를 결정하기 위한 변수
     
     # ul[0], ul[1]은 원의 좌측 상단 모서리 위치
     ul = [int(pt[0] - tmpSize), int(pt[1]-tmpSize)]
     
     # br은 pt를 중심으로 3sigma만끔 떨어진 위치의 좌표
     # br은 우측 하단 꼭지점의 좌표로 br[0]<-x좌표, br[1]<-y좌표
     br = [int(pt[0] + tmpSize +1), int(pt[1] + tmpSize +1)]
     
     #img.shape[0]은 이미지 가로 크기, img.shape[1]은 세로 크기
     if(ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
        br[0] < 0 or br[1] < 0):
          return to_torch(img)
     
     # Generate gaussian
     size = 2 * tmpSize + 1
     x = np.arange(0,size,1,float)
     # newaxis는 new axis를 추가하여 배열의 모양을 바꿈
     y = x[:, np.newaxis]#<- 1차원 x로 2차원 열벡터가 만들어짐
     x0 = y0 = size//2
     sigma = size /4.0
     
     # The gaussian is not normalized, we want the center value to equal 1
     # 2차원 가우시안 분포 함수
     """
          중심점에서 멀어질수록 값이 작아지며, 중심점에서 가장 높은 값으로
          아래 코드는 가우시안 마스크를 생성
     """
     g = np.exp(- ((x-x0) **2 + (y-y0)**2) / (2 * sigma **2))
     g[g>0] = 1
     
     # 가우시안 마스크 g를 적용할 수 있는 이미지 범위
     # g_x의 첫 번째는 가우시안 마스크가 이미지 왼쪽 경계를 벗어나지 않도록 0과 ul[0] 중 큰 값
     # g_x의 두 번째는 가우시안 마스크가 이미지 오른쪽 경계를 벗어나지 않도록 br[0]와 이미지 폭 중 작은 값
     g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
     g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
     
     # Image range
     img_x = max(0, ul[0]), min(br[0], img.shape[1])
     img_y = max(0, ul[1]), min(br[1], img.shape[0])

     img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
     return to_torch(img)

# 2D 좌표 평면에서 a,b 두 점이 있을 때, 선분 ab를 이루는
# 직각삼각형의 나머지 한 꼭짓점을 찾아내는 함수
def get_3rd_point(a, b):
     direct = a - b # a,b의 벡터
     # y축 방향으로 회전한 벡터를 구하고 b를 더하면 나머지 꼭짓점이 됨
     return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def cropBox(img, ul, br ,resH, resW):
     """
     Args:
         img: 자를 이미지 
         ul: 자를 이미지의 왼쪽 상단 좌표
         br: _자를 이미지의 오른쪽 하단 좌표
         resH: 자를 이미지의 높이
         resW: 자를 이미지의 너비

     Returns:
         _type_: 자른 이미지를 torch.Tensor로 반환
         
     """
     ul = ul.int()
     br = (br-1).int()
     
     # crop size 계산
     # br[0]은 우측 끝 x좌표, br[1]은 우측 끝 y좌표
     # ul[0]은 좌측 끝 x좌표, ul[1]은 좌측 끝 y좌표
     lenH = max((br[1]-ul[1]).item(), (br[0]-ul[0]).item()*resH/resW)
     lenW = lenH * resW / resH
     if img.dim() == 2:
          img = img[np.newaxis, :]
     
     # 새롭게 크롭할 이미지의 형태 [1]은 높이 [0]은 너비     
     box_shape = [(br[1] - ul[1]).item(), (br[0]-ul[0]).item()]
     # crop_image를 원하는 크기로 만들기 위해 padding size
     pad_size = [(lenH - box_shape[0]) // 2, (lenW - box_shape[1]) //2]
     
     #padding Zeros
     # 크롭할 박스가 이미지 경계를 넘어갈수(잘릴수) 있으므로, 
     # 이미지 경계를 벗어나는 부분은 0으로 패딩
     
     if ul[1] > 0: #image의 상단 경계를 벗어나지 않도록 설정
          img[:, :ul[1], :]= 0 # 상단을 0으로 padding
     if ul[0] > 0:  #image의 왼쪽 경계를 벗어나지 않도록 설정
          img[:, :, :ul[0]] = 0 # 왼쪽을 0으로 padding
     if br[1] < img.shape[1]-1: #image의 하단 경계를 벗어나지 않도록 설정
          img[:, br[1] +1:, :]= 0 # 하단을 0으로 padding
     if br[0] < img.shape[2] -1: #image의 오른쪽 경계를 벗어나지 않도록 설정
          img[:, :, br[0] + 1:]=0# 오른쪽을 0으로 padding
          
     src = np.zeros((3,2),dtype=np.float32) #원본 이미지에서 자를 영역
     dst = np.zeros((3,2), dtype=np.float32) #새로운 이미지에서 붙여넣을 영역
     
     #잘라낸 왼쪽 상단 좌표, pad_size 빼는 이유는 적용한 padding 때문                      
     src[0,:] = np.array(
          [ul[0] - pad_size[1], ul[1] - pad_size[0]],np.float32)
     
     #잘라낼 영역의 우측 하단 좌표, 적용한 padding만큼 이동시켜야해서
     src[1,:] = np.array(
          [br[0] + pad_size[1], br[1] + pad_size[0]], np.float32)
     
     # 붙여넣을 대상 image에서 잘라낼 영역의 좌측 상단 모서리 (0,0)
     dst[0, :] = 0
     
     # 붙여넣을 대상 image에서 잘라낼 영역의 우측 하단 모서리 (resW-1,resH-1)
     dst[1, :] = np.array([resW - 1, resH - 1], np.float32)
     
       
     src[2:, :] = get_3rd_point(src[0, :], src[1, :])
     dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])
     
     trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

     dst_img = cv2.warpAffine(torch_to_im(img), trans,
                             (resW, resH), flags=cv2.INTER_LINEAR)

     # 자른 이미지를 torch.Tensor로 반환
     return im_to_torch(torch.Tensor(dst_img))

#관절 17개 사용
def drawCOCO(inps,preds,scores):
     """
     Args:
         inps: input으로 (N,C,H,W)를 받음
         preds: (N,17,2) 텐서로 17개의 사람 관절의 x,y 좌표
         scores: 각 뼈대 좌표의 점수(신뢰도)

     Returns:
         _type_: _description_
     """
     assert inps.dim() == 4 # 차원이 4인지 검사
     
     #key point 17개의 각 색상 첫 번째 부터 green,blue, ...
     p_color = ['g', 'b', 'purple', 'b', 'purple',
                'y', 'orange', 'y', 'orange', 'y', 'orange',
                'pink', 'r', 'pink', 'r', 'pink', 'r']
     
     nImg = inps.size(0)
     imgs=[]
     for n in range(nImg):
          img = to_numpy(inps[n])
          img = np.transpose(img, (1,2,0))
          imgs.append(img)
          
     fig = plt.figure()
     plt.imshow(imgs[0])
     ax = fig.add_subplot(1,1,1)
     
     # 17개의 관절을 순회하며 일정 score이상만 추가
     for p in range(17):
          # score가 0.2보다 작으면 그리지 않음
          if scores[0][p][0] < 0.2:
               continue
          x, y = preds[0][p]
          cor = (round(x), round(y)),3
          ax.add_patch(plt.Circle(*cor,color=p_color[p]))
     plt.axis('off')
     
     plt.show()
     return imgs

# drawCOCO와는 데이터셋이 다르고
# 관절 16개 사용 
def drawMPII(inps,preds):
     assert inps.dim() == 4
     p_color = ['g', 'b', 'purple', 'b', 'purple',
               'y', 'o', 'y', 'o', 'y', 'o',
               'pink', 'r', 'pink', 'r', 'pink', 'r']
     
     p_color = ['r', 'r', 'r', 'b', 'b', 'b',
               'black', 'black', 'black', 'black',
               'y', 'y', 'white', 'white', 'g', 'g']
     
     nImg = inps.size(0)
     imgs=[]
     for n in range(nImg):
          img = to_numpy(inps[n])
          img = np.transpose(img, (1, 2, 0))
          imgs.append(img)
     fig = plt.figure()
     plt.imshow(imgs[0])
     ax = fig.add_subplot(1, 1, 1)
     #print(preds.shape)
     for p in range(16):
        x, y = preds[0][p]
        cor = (round(x), round(y)), 10
        ax.add_patch(plt.Circle(*cor, color=p_color[p]))
     plt.axis('off')

     plt.show()

     return imgs


# drawCircle는 새 이미지를 만들어 적용하지만
# 이 함수는 기본이미지에 그대로 수정함
# g의 범위가 다름
def drawBigCircle(img, pt sigma):
     img = to_numpy(img)
     tmpSize = 3 * sigma
     
     ul = [int(pt[0] - tmpSize), int(pt[1] - tmpSize)]
     br = [int(pt[0] + tmpSize + 1), int(pt[1] + tmpSize + 1)]
     
     if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
          br[0] < 0 or br[1] < 0):
          # If not, just return the image as is
          return to_torch(img)
     
     size = 2 * tmpSize + 1
     x = np.arange(0, size, 1, float)
     y = x[:, np.newaxis]
     x0 = y0 = size // 2
     sigma = size / 4.0
     # The gaussian is not normalized, we want the center value to equal 1
     g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
     g[g > 0.4] = 1
     # Usable gaussian range
     g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
     g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
     # Image range
     img_x = max(0, ul[0]), min(br[0], img.shape[1])
     img_y = max(0, ul[1]), min(br[1], img.shape[0])

     img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
     return to_torch(img)

# 마찬가지로 g의 범위만 다름
def drawSmallCircle(img, pt, sigma):
     img = to_numpy(img)
     tmpSize = 3 * sigma
     # Check that any part of the gaussian is in-bounds
     ul = [int(pt[0] - tmpSize), int(pt[1] - tmpSize)]
     br = [int(pt[0] + tmpSize + 1), int(pt[1] + tmpSize + 1)]

     if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
          # If not, just return the image as is
          return to_torch(img)

     # Generate gaussian
     size = 2 * tmpSize + 1
     x = np.arange(0, size, 1, float)
     y = x[:, np.newaxis]
     x0 = y0 = size // 2
     sigma = size / 4.0
     # The gaussian is not normalized, we want the center value to equal 1
     g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
     g[g > 0.5] = 1
     # Usable gaussian range
     g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
     g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
     # Image range
     img_x = max(0, ul[0]), min(br[0], img.shape[1])
     img_y = max(0, ul[1]), min(br[1], img.shape[0])   

     img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
     return to_torch(img)


def drawGaussian(img, pt, sigma):
     img = to_numpy(img)
     tmpSize = 3 * sigma
     # Check that any part of the gaussian is in-bounds
     ul = [int(pt[0] - tmpSize), int(pt[1] - tmpSize)]
     br = [int(pt[0] + tmpSize + 1), int(pt[1] + tmpSize + 1)]

     if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
          br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
          return to_torch(img)

    # Generate gaussian
     size = 2 * tmpSize + 1
     x = np.arange(0, size, 1, float)
     y = x[:, np.newaxis]
     x0 = y0 = size // 2
     sigma = size / 4.0    
     # The gaussian is not normalized, we want the center value to equal 1
     g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

     # Usable gaussian range
     g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
     g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
     # Image range
     img_x = max(0, ul[0]), min(br[0], img.shape[1])
     img_y = max(0, ul[1]), min(br[1], img.shape[0])

     img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
     return to_torch(img)


# 좌우로 뒤집는 함수 -> 데이터 증강을 위해
def shuffleLR(x, dataset):
     # flipRefsms 데이터의 좌우 반전을 어떤 축으로 진행할지 저장
     flipRef = dataset.flipRef 
     assert (x.dim() == 3 or x.dim() ==4)
     
     for pair in flipRef:
          dim0, fim1 = pair
          dim0 -= 1 # 1을 빼는 이유는 tensor는 0부터 시작하기 때문
          dim1 -= 1
          
          # x의 차원 수에 따라 x를 좌우 반전 시킴
          if x.dim() == 4: # [:,dim1]과 [:,dim0]을 교환
               tmp = x[:, dim1].clone()
               x[:, dim1] = x[:, dim0].clone()
               x[:, dim0] = tmp.clone()
          else: # dim1과 dim0을 교환
               tmp = x[dim1].clone()
               x[dim1] = x[dim0].clone()
               x[dim0] = tmp.clone()
          # clone을 쓰는 이유는 tensor는 메모리를 공유하기 때문에 복제 후 교환     
     return x


# 3D 또는 4D 텐서를 뒤집는 함수
def flip(x):
     assert (x.dim() == 3 or x.dim() == 4)
     dim = x.dim() - 1
     if '0.4.1' in torch.__version__ or '1.0' in torch.__version__:
          return x.flip(dims=(dim,))
     else:
          is_cuda = False
          if x.is_cuda:
               is_cuda = True
               x = x.cpu() 
          x = x.numpy().copy() # deep copy
          if x.ndim == 3:
          # fliplr은 2차원 배열을 좌우로 뒤집어 반환하는 함수
               x = np.transpose(np.fliplr(np.transpose(x, (0, 2, 1))), (0, 2, 1))
          elif x.dim == 4:
               for i in range(x.shape[0]):
                    x[i] = np.transpose(
                         np.fliplr(np.transpose(x[i], (0,2,1)), (0, 2, 1)))
          
          x = torch.from_numpy(x.copy())
          if is_cuda:
               x = x.cuda()
          return x


# bounding box를 잘라내어 크기가 동일한 새로운 이미지로 만드는 함수
def crop_dets(img, boxes, height, width):
     """_summary_

     Args:
         img: 이미지
         boxes: N개의 bounding box 좌표를 [x1,y1,x2,y2]형태로 저장
         height: 자른 이미지 높이
         width: 자른 이미지 너비

     Returns:
         각 bounding box의 좌상단,우하단,자른 이미지를 return
     """
     img = im_to_torch(img) # tensor로 변환
     img_h = img.size(1) #원래 이미지 높이
     img_w = img.size(2) #원래 이미지 너비
     # 이미지 채널보다 아래 값들을 뺀다
     img[0].add_(-0.406)
     img[1].add_(-0.457)
     img[2].add_(-0.480)
               
     inps = torch.zeros(len(boxes), 3, height, width)
     pt1 = torch.zeros(len(boxes), 2)
     pt2 = torch.zeros(len(boxes), 2)          
               
     for i,box in enumerate(boxes):
          # bounding box마다 좌상단, 우하단의 좌표를 구함
          upLeft = torch.Tensor((float(box[0]), float(box[1])))
          bottomRight = torch.Tensor((float(box[2]), float(box[3])))         
          
          # 박스의 너비와 높이를 계산
          h = bottomRight[1] - upLeft[1] 
          w = bottomRight[0] - upLeft[0]
          if w > 100:
               scaleRate = 0.2
          else:
               scaleRate = 0.3

          # 박스를 scaleRate에 따라 축소시켜 bounding box의 좌표를 수정
          upLeft[0] = max(0, upLeft[0] - w * scaleRate / 2)
          upLeft[1] = max(0, upLeft[1] - h * scaleRate / 2)
          bottomRight[0] = max(min(img_w - 1, bottomRight[0] + w * scaleRate / 2), upLeft[0] + 5)
          bottomRight[1] = max(min(img_h - 1, bottomRight[1] + h * scaleRate / 2), upLeft[1] + 5)

          # cropBox 함수를 이용해 이미지에서 box영역을 자름
          inps[i] = cropBox(img.clone(), upLeft, bottomRight, height, width)
          pt1[i] = upLeft
          pt2[i] = bottomRight

     # 좌상단, 우하단, 자른 박스 영역을 return
     return inps, pt1, pt2


# 점을 주어진 각도만큼 회전시킨 후의 좌표 계산
def get_dir(src_point, rot_rad):
     # sin과 cos을 이용함
     sn, cs = np.sin(rot_rad), np.cos(rot_rad)

     src_result = [0, 0]
     src_result[0] = src_point[0] * cs - src_point[1] * sn
     src_result[1] = src_point[0] * sn + src_point[1] * cs

     return src_result



# 히트맵(hm)과 후보 키포인트(candidate_points)를 입력받아 최종적으로 검출된 키포인트 반환
def processPeaks(candidate_points, hm, pt1, pt2, inpH, inpW, resH, resW):
#type:(Tensor,Tensor,Tensor,Tensor,float,float,float,float)->List[Tensor]


     # 후보 키포인트가 비어있을 때
     if candidate_points.shape[0] == 0:
          maxval = np.max(hm.reshape(1, -1), 1)
          idx = np.argmax(hm.reshape(1, -1), 1)
          
          x = idx % resW
          y = int(idx / resW)
          
          candidate_points = np.zeros((1,3))
          candidate_points[0, 0:1] = x
          candidate_points[0, 1:2] = y
          candidate_points[0, 2:3] = maxval         
          
     res_pts = []
     for i in range(candidate_points.shape[0]):
          x, y, maxval = candidate_points[i][0], candidate_points[i][1],candidate_points[i][2]
          
          # 감지된 point가 존재하고 maxval이 너무 작아 새로운 point를 찾을 필요 X
          if bool(maxval < 0.05) and len(res_pts) > 0:
               pass
          else:
               if bool(x > 0) and bool(x < resW - 2):
                    if bool(hm[int(y)][int(x) + 1] - hm[int(y)][int(x) - 1] > 0):
                         x += 0.25
                    elif bool(hm[int(y)][int(x) + 1] - hm[int(y)][int(x) - 1] < 0):
                         x -= 0.25
               if bool(y > 0) and bool(y < resH - 2):
                    if bool(hm[int(y) + 1][int(x)] - hm[int(y) - 1][int(x)] > 0):
                         y += (0.25 * inpH / inpW)
                    elif bool(hm[int(y) + 1][int(x)] - hm[int(y) - 1][int(x)] < 0):
                         y -= (0.25 * inpH / inpW)

               #pt = torch.zeros(2)
               pt = np.zeros(2)
               pt[0] = x + 0.2
               pt[1] = y + 0.2

               pt = transformBoxInvert(pt, pt1, pt2, inpH, inpW, resH, resW)

               res_pt = np.zeros(3)
               res_pt[:2] = pt
               res_pt[2] = maxval

               res_pts.append(res_pt)

               if maxval < 0.05:
                    break
     return res_pts


# 특정 값을 가지는 최대값을 찾아서 해당 위치와 값을 포함하는 리스트 반환
def findPeak(hm):
     # 5x5 필터를 적용해 heatmap 내 최댓값 구함
     mx = maximum_filter(hm, size=5)
     
     # mx와 hm이 같고, hm의 값이 0.1보다 큰 위치의 index를 찾음
     # zip 함수를 사용하여 인덱스를 (y,x) 형태로 묶은 후 반환
     idx = zip(*np.where((mx == hm) * (hm > 0.1)))
     
     candidate_points = []
     for (y, x) in idx:
          candidate_points.append([x, y, hm[y][x]])
          
     if len(candidate_points) == 0:
          return torch.zeros(0)
     candidate_points = np.array(candidate_points)
     candidate_points = candidate_points[np.lexsort(-candidate_points.T)]
     return torch.Tensor(candidate_points)



# parameter: 이미지, 회전 각도, 가로 크기, 세로 크기
def cv_rotate(img, rot, resW, resH):
     center = np.array((resW - 1, resH -2)) / 2 # 회전 중심점을 설정
     rot_rad = np.pi * rot / 180                # 회전 각도를 변환
     
     # get_dir을 통해 회전 변환에 사용할 좌표 구함
     src_dir = get_dir([0, (resH - 1) * -0.5], rot_rad)
     dst_dir = np.array([0, (resH-1)*- 0.5], np.float32)
     
     src = np.zeros((3,2), dtype=np.float32)
     dst = np.zeros((3,2), dtype=np.float32)
     
     src[0, :] = center
     src[1, :] = center + src_dir
     dst[0, :] = [(resW - 1) * 0.5, (resH - 1) * 0.5]
     dst[1, :] = np.array([(resW - 1) * 0.5, (resH - 1) * 0.5]) + dst_dir

     src[2:, :] = get_3rd_point(src[0, :], src[1, :])
     dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

     # getAffineTransform을 이용해 src와 dst의 좌표를 대응시켜 회전,이동 변환행렬을 구함
     trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

     #
     dst_img = cv2.warpAffine(torch_to_im(img), trans,
                             (resW, resH), flags=cv2.INTER_LINEAR)

     return im_to_torch(torch.Tensor(dst_img))
     


def transformBoxInvert_batch(pt, ul, br, inpH, inpW, resH, resW):
     """
     pt:       [n, 17, 2]
     ul:       [n, 2]
     br:       [n, 2]
     """
     
     num_pt = pt.shape[1]
     center = (br -1 -ul) / 2
     
     size = br - ul
     size[:, 0] *= 