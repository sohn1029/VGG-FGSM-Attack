import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split

import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import numpy as np
import matplotlib.pyplot as plt

from model.vgg16 import VGG16



if __name__ == '__main__':

    # gpu 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 생성
    model = VGG16(_input_channel=1, num_class=10)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # 테스트 할 모델 체크포인트 경로
    test_checkpoint = './checkpoints/robust5_checkpoint.pth'

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
    # MNIST dataset

    # 테스트 데이터셋 로드
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Normalize((0.5), (0.5))
    ])
    test_data = dsets.MNIST(root='data/', train=False, transform=transform, download=True)
    #test_data  = dsets.MNIST(root='data/', train=False, transform=transforms.ToTensor(), download=True)
    #test_data = data_utils.Subset(test_data, torch.arange(3000))
    # 데이터로더 생성
    test_loader = torch.utils.data.DataLoader(
                    dataset=test_data,
                    batch_size=batch_size,
                    shuffle=False)
    print(len(test_loader))
    #------------------------------------------

    test_loss = 0.0
    test_accuracy = 0.0
    adv_examples = []
    model.eval()
    with torch.no_grad():
        for x_b, target_b in test_loader:
            for index in range(batch_size):
                if index >= len(target_b):
                    break
                x = x_b[index].unsqueeze(dim = 0)
                target = target_b[index].unsqueeze(dim = 0)
                x = x.to(device)
                target = target.to(device)

                x.requires_grad = True

                out = model(x)

                # loss 계산
                loss = criterion(out, target)
                test_loss = test_loss + loss.item()

                # accuracy 계산
                output_label = torch.argmax(out, 1).to('cpu').numpy()
                target_label = target.to('cpu').numpy()

                correct_num = 0.0

                
                answer = target_label
                predict = output_label

                if answer == predict:
                    correct_num = correct_num + 1.0


                accuracy = correct_num / batch_size
                test_accuracy = test_accuracy + accuracy



        avg_test_loss = test_loss / len(test_loader)
        avg_test_accuracy = test_accuracy / len(test_loader)

        print('test loss : %f' % (avg_test_loss))
        print('test accuracy : %f' % (avg_test_accuracy))