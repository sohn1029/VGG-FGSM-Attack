import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split

import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from model.vgg16 import VGG16
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
def load_image(path:str):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

if __name__ == '__main__':

    # gpu 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 생성
    model = VGG16(_input_channel=1, num_class=10)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # 테스트 할 모델 경로
    test_checkpoint = 'checkpoints/robust5_checkpoint.pth'

    #------------------------------------------
    # 이전 체크포인트로부터 모델 로드
    print("start model load...")
    # 체크포인트 로드
    checkpoint = torch.load(test_checkpoint, map_location=device)

    # 각종 파라미터 로드
    model.load_state_dict(checkpoint['model'])
    batch_size = checkpoint['batch_size']

    print("model load end.")
    #------------------------------------------

    #------------------------------------------
    # 테스트할 이미지 로드
    #img_path = "data/test_img/test_img_4.jpg"
    img_path = "data/test_img/mnistimg3.png"
    test_img = load_image(img_path)
    plt.imshow(test_img, cmap='gray')
    plt.show()

    # 이미지를 그레이스케일 및 텐서로 변환
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    test_img = transform(test_img)
    #------------------------------------------
    plt.imshow(test_img[0])
    plt.show()
    # 결과 추론
    model.eval()

    test_input_img = test_img.unsqueeze(0)
    test_input_img = test_input_img.to(device)
    output = model(test_input_img)

    # 추론 결과
    print("0~9 숫자별 확률:", output.cpu().detach().numpy())
    print("최종 예측: ", torch.argmax(output, 1).to('cpu').numpy())