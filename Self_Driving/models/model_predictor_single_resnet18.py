import torch
from torch import nn
import torchvision.models as models

class Predictor_single(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3)

        resnet18 = models.resnet18(pretrained=False)
        lst_res18 = list(resnet18.children())

        self.res1 =  torch.nn.Sequential(*(lst_res18[1:4]))
        self.res2 =  torch.nn.Sequential(*(lst_res18[4:5]))
        self.res3 =  torch.nn.Sequential(*(lst_res18[5:6]))
        self.res4 =  torch.nn.Sequential(*(lst_res18[6:7]))
        self.res5 =  torch.nn.Sequential(*(lst_res18[7:8]))
        self.avpool2d = torch.nn.Sequential(*(lst_res18[8:9]))

        self.fc1 = nn.Linear(512, 128)

        self.fc_pred = nn.Linear(128, 1)

        self.softplus = nn.Softplus()

        self.flatten = nn.Flatten()

    def forward(self, image):
        # Input size for now: 5 stacked frames and the lateral speed of first 4 frames, want to predict the lateral speed of 5th frame

        feature1 = self.conv1(image)
        feature2 = self.res1(feature1)
        feature3 = self.res2(feature2)
        feature4 = self.res3(feature3)
        feature5 = self.res4(feature4)
        feature6 = self.res5(feature5)
        out = self.avpool2d(feature6).squeeze()

        dis_pred = self.fc1(out)
        dis_pred = self.fc_pred(dis_pred)

        dis_pred = self.softplus(dis_pred)
        
        return dis_pred
