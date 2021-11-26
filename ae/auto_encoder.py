
import torch.nn as nn
from torch.distributions.normal import Normal
import functools
import operator
import torch.nn.functional as F
import torch
import random
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import cv2
import os, random
from PIL import Image
import numpy as np

def add_noise(images, mean=0, std=0.1):
    normal_dst = Normal(mean, std)
    noise = normal_dst.sample(images.shape)
    noisy_image = noise + images
    return noisy_image

class Auto_Encoder_Model_PReLu(nn.Module):
    def __init__(self):
        super(Auto_Encoder_Model_PReLu, self).__init__()
        self.prelu = nn.PReLU() 
        # Encoder
        self.conv1 = nn.Conv2d(3, 128, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2) # 16*8*17

        # Decoder
        self.tran_conv1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, output_padding=[1, 0])
        self.tran_conv2 = nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, output_padding=[1, 0])
        self.tran_conv3 = nn.ConvTranspose2d(128, 3, kernel_size=8, stride=4)
        

    def forward_pass(self, x):
        output = self.prelu(self.conv1(x))
        output = self.prelu(self.conv2(output))
        output = self.prelu(self.conv3(output))
        return output

    def reconstruct_pass(self, x):
        output = self.prelu(self.tran_conv1(x))
        output = self.prelu(self.tran_conv2(output))
        output = self.prelu(self.tran_conv3(output))
        return output

    def forward(self, x):
        output = self.forward_pass(x)
        output = self.reconstruct_pass(output)
        return output

class Auto_Encoder_Model_PReLu224(nn.Module):
    def __init__(self):
        super(Auto_Encoder_Model_PReLu224, self).__init__()
        self.prelu = nn.PReLU() 
        # Encoder
        self.conv1 = nn.Conv2d(3, 128, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2) # 16*8*17

        # Decoder
        self.tran_conv1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2)
        self.tran_conv2 = nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, output_padding=[1, 1])
        self.tran_conv3 = nn.ConvTranspose2d(128, 3, kernel_size=8, stride=4)
        

    def forward_pass(self, x):
        output = self.prelu(self.conv1(x))
        output = self.prelu(self.conv2(output))
        output = self.prelu(self.conv3(output))
        return output

    def reconstruct_pass(self, x):
        output = self.prelu(self.tran_conv1(x))
        output = self.prelu(self.tran_conv2(output))
        output = self.prelu(self.tran_conv3(output))
        return output

    def forward(self, x):
        output = self.forward_pass(x)
        output = self.reconstruct_pass(output)
        return output

def train():
    #train AE
    device = 'cuda:0'
    model = Auto_Encoder_Model_PReLu224().to(device)
    # model = Spatial_Auto_Encoder_Model(device=device).to(device)
    model.load_state_dict(torch.load('./ae/ae_full_prelu_224.pth'))
    ae_criterion = torch.nn.MSELoss().to(device)
    optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()))

    #load data
    # image = torch.from_numpy(image).float()/255.0 
    # ToTensor already divided the image by 255
    batch_size = 16
    dataset = datasets.ImageFolder('/home/tinvn/TIN/NLP_RL_Code/data/data_ae/', transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    #train
    epochs = 100
    for epoch in range(0, epochs):
        current_loss = 0
        for i,data in enumerate(data_loader,0):
            input, _ = data
            input = input.to(device)
            
            optim.zero_grad()
            encoder = model.forward_pass(input)
            decoder = model.reconstruct_pass(encoder)
            loss = ae_criterion(decoder, input)

            loss.backward()
            optim.step()

            loss = loss.item()
            current_loss += loss
        print('{} loss : {}'.format(epoch+1, batch_size*current_loss/len(data_loader)))

    torch.save(model.state_dict(),'./ae/ae_full_prelu_224.pth')

def test():
    device = 'cuda:0'
    model = Auto_Encoder_Model_PReLu224().to(device)
    weight_path = './ae/ae_full_prelu_224.pth'
    print('load initial weights CDAE from: %s'%(weight_path))
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    folder = '/home/tinvn/TIN/NLP_RL_Code/data/data_ae/images/'
    files = os.listdir(folder)
    f = random.choice(files)

    image_path = folder + f
    # image_path = "/home/tinvn/Desktop/14167.png"
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_pil = Image.fromarray(image)
    image = image/255
    # print(image)
    
    
    # to_tensor = transforms.ToTensor()
    image = torch.Tensor(image)
    image = image.permute(-1,0,1)
    image = image.unsqueeze(0)
    
    t = time.time()
    
    image = image.to(device)
    encoder = model.forward_pass(image)
    decoder = model.reconstruct_pass(encoder)
    decoder = decoder.squeeze()
    decoder = decoder.permute(1, 2, 0)
    
    f = plt.figure()
    f.add_subplot(1,1, 1) #1,2,1
    plt.imshow(decoder.cpu().detach().numpy())

    # f.add_subplot(1,2, 2)
    # plt.imshow(image_pil)

    print(time.time()-t)
    plt.show()

if __name__ == "__main__":
    train()
    
    

