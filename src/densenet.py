import torch
import torch.nn as nn
import torchvision

class WPAL_dense201(nn.Module):
    def __init__(self):
        super(WPAL_dense201, self).__init__()
        model = torchvision.models.densenet201(pretrained=True)
        
        #OUT1-> 512*28*28
        #OUT2-> 1792*14*14
        #OUT3-> 1920*7*7

        return_layers = {'denseblock2': 'out1', 'denseblock3': 'out2', 'denseblock4': 'out3'}
        self.backbone = torchvision.models._utils.IntermediateLayerGetter(model.features, return_layers)
        
        ## CONV1_E Head1
        self.Conv1_E = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fssp1_l1 = nn.AdaptiveMaxPool2d((1, 1))
        self.fssp1_l2 = nn.AdaptiveMaxPool2d((3, 1))
        self.FC1_E = nn.Linear(2048, 512)
        self.dr_1 = nn.Dropout(0.5)

        ## CONV2_E Head2
        self.Conv2_E = nn.Conv2d(1792, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fssp2_l1 = nn.AdaptiveMaxPool2d((1, 1))
        self.fssp2_l2 = nn.AdaptiveMaxPool2d((3, 3))
        self.FC2_E = nn.Linear(5120, 512)
        self.dr_2 = nn.Dropout(0.5)

        ## CONV3_E Head3
        self.Conv3_E = nn.Conv2d(1920, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  
        self.fssp3_l1 = nn.AdaptiveMaxPool2d((1, 1))
        self.fssp3_l2 = nn.AdaptiveMaxPool2d((3, 1))
        self.FC3_E = nn.Linear(4096, 1024)
        self.dr_3 = nn.Dropout(0.5)

        ## Concatenation
        self.FC_SYN1 = nn.Linear(2048, 2048)
        self.dr_4 = nn.Dropout(0.5)
        self.fc = nn.Linear(2048, 1000)
        

    def forward(self, x):
        out1 = self.backbone(x)['out1']
        out2 = self.backbone(x)['out2']
        out3 = self.backbone(x)['out3']

        ## CONV1_E Head1
        out1 = self.Conv1_E(out1)
        out1_l1 = self.fssp1_l1(out1).view(out1.size(0), -1)
        out1_l2 = self.fssp1_l2(out1).view(out1.size(0), -1)
        out1 = torch.cat([out1_l1, out1_l2], dim=1)
        out1 = self.FC1_E(out1)
        out1 = self.dr_1(out1)

        ## CONV2_E Head2
        out2 = self.Conv2_E(out2)
        out2_l1 = self.fssp2_l1(out2).view(out2.size(0), -1)
        out2_l2 = self.fssp2_l2(out2).view(out2.size(0), -1)
        out2 = torch.cat([out2_l1, out2_l2], dim=1)
        out2 = self.FC2_E(out2)
        out2 = self.dr_2(out2)

        ## CONV3_E Head3
        out3 = self.Conv3_E(out3)
        out3_l1 = self.fssp3_l1(out3).view(out3.size(0), -1)
        out3_l2 = self.fssp3_l2(out3).view(out3.size(0), -1)
        out3 = torch.cat([out3_l1, out3_l2], dim=1)
        out3 = self.FC3_E(out3)
        out3 = self.dr_3(out3)

        ## Concatenation
        out = torch.cat([out3, out2, out1], dim=1)
        out = self.FC_SYN1(out)
        out = self.dr_4(out)
        out = self.fc(out)

        return out
