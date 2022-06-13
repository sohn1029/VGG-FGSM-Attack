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

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad
    return perturbed_image

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
                    shuffle=True)
    #------------------------------------------
    print(len(test_loader))
    test_loss = 0.0
    test_accuracy = 0.0

    model.eval()
    def test( model, device, test_loader, epsilon ):
    
    # 정확도 카운터
        correct = 0
        adv_examples = []
        err = 0
        # 테스트 셋의 모든 예제에 대해 루프를 돕니다
        for data_b, target_b in test_loader:
            for index in range(batch_size):
                if index >= len(target_b):
                    break
                data = data_b[index].unsqueeze(dim = 0)
                target = target_b[index].unsqueeze(dim = 0)
         
                # 디바이스(CPU or GPU) 에 데이터와 라벨 값을 보냅니다
                data, target = data.to(device), target.to(device)

                # 텐서의 속성 중 requires_grad 를 설정합니다. 공격에서 중요한 부분입니다
                data.requires_grad = True

                # 데이터를 모델에 통과시킵니다
                output = model(data)
                init_pred = output.max(1, keepdim=True)[1] # 로그 확률의 최대값을 가지는 인덱스를 얻습니다

                # 만약 초기 예측이 틀리면, 공격하지 않도록 하고 계속 진행합니다
                if init_pred.item() != target.item():
                    err = err + 1
                    continue

                # 손실을 계산합니다
                loss = criterion(output, target)

                # 모델의 변화도들을 전부 0으로 설정합니다
                #model.zero_grad()

                # 후방 전달을 통해 모델의 변화도를 계산합니다
                loss.backward()

                # 변화도 값을 모읍니다
                data_grad = data.grad.data

                # FGSM 공격을 호출합니다
                perturbed_data = fgsm_attack(data, epsilon, data_grad)
                # 작은 변화가 적용된 이미지에 대해 재분류합니다
                output = model(perturbed_data)
                #output = model(data)
                
                # 올바른지 확인합니다
                final_pred = output.max(1, keepdim=True)[1] # 로그 확률의 최대값을 가지는 인덱스를 얻습니다
                # print(" final : ",final_pred.item())
                # print("target : ",target.item())
                if final_pred.item() == target.item():
                    correct += 1
                    # 0 엡실론 예제에 대해서 저장합니다
                    if (epsilon == 0) and (len(adv_examples) < 5):
                        adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                        adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
                else:
                    # 추후 시각화를 위하 다른 예제들을 저장합니다
                    if len(adv_examples) < 5:
                        adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                        adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        print('err :', err)
        # 해당 엡실론에서의 최종 정확도를 계산합니다
        final_acc = correct/float(len(test_loader)*batch_size)
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader)*batch_size, final_acc))

        # 정확도와 적대적 예제를 리턴합니다
        return final_acc, adv_examples

    epsilons = [0.0, 0.05, 0.1, 0.2, 0.3]
    accuracies = []
    examples = []

    # 각 엡실론에 대해 테스트 함수를 실행합니다
    for eps in epsilons:
        acc, ex = test(model, device, test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)

    plt.figure(figsize=(5,5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()

    cnt = 0
    plt.figure(figsize=(8,10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons),len(examples[0]),cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig,adv,ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()
