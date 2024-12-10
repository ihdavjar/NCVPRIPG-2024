import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F



class head_1(nn.Module):
    def __init__(self, in_features):
        super(head_1, self).__init__()

        # self.regularize_dp = nn.Dropout(0.5)

        # For Upper body color
        # self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1_1 = nn.Linear(in_features, 9)

        # For Lower body color
        # self.fc2 = nn.Linear(in_features, hidden_features)
        self.fc2_1 = nn.Linear(in_features, 9)

        # For Upper body clothing
        # self.fc3 = nn.Linear(in_features, hidden_features)
        self.fc3_1 = nn.Linear(in_features, 8)

        # For Lower body clothing
        # self.fc4 = nn.Linear(in_features, hidden_features)
        self.fc4_1 = nn.Linear(in_features, 7)

        # For Sleeve length
        # self.fc5 = nn.Linear(in_features, hidden_features)
        self.fc5_1 = nn.Linear(in_features, 3)

        # For Carry item
        # self.fc6 = nn.Linear(in_features, hidden_features)
        self.fc6_1_1 = nn.Linear(in_features, 1)
        self.fc6_1_2 = nn.Linear(in_features, 1)
        self.fc6_1_3 = nn.Linear(in_features, 1)
        
        # For Head wear
        # self.fc7 = nn.Linear(in_features, hidden_features)
        self.fc7_1 = nn.Linear(in_features, 2)

        # For Foot wear
        # self.fc8 = nn.Linear(in_features, hidden_features)
        self.fc8_1 = nn.Linear(in_features, 4)

        # For Pose
        # self.fc9 = nn.Linear(in_features, hidden_features)
        self.fc9_1 = nn.Linear(in_features, 3)

        # For View
        # self.fc10 = nn.Linear(in_features, hidden_features)
        self.fc10_1 = nn.Linear(in_features, 3)
        
    def forward(self, x):

        x = x.view(x.size(0), -1)

        # x = self.regularize_dp(x)

        out1 = self.fc1_1(x)
        out2 = self.fc2_1(x)
        out3 = self.fc3_1(x)
        out4 = self.fc4_1(x)
        out5 = self.fc5_1(x)
        
        out6 = self.fc6_1_1(x)
        out7 = self.fc6_1_2(x)
        out8 = self.fc6_1_3(x)
        
        out9 = self.fc7_1(x)
        out10 = self.fc8_1(x)
        out11 = self.fc9_1(x)

        out12 = self.fc10_1(x)
        
        # Concatenating the outputs
        out = torch.cat([out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12], dim=1)
        return out


class head_2(nn.Module):
    def __init__(self, in_features, hidden_features=512):
        super(head_2, self).__init__()

        # For Upper body color
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1_1 = nn.Linear(hidden_features, 9)

        # For Lower body color
        self.fc2 = nn.Linear(in_features, hidden_features)
        self.fc2_1 = nn.Linear(hidden_features, 9)

        # For Upper body clothing
        self.fc3 = nn.Linear(in_features, hidden_features)
        self.fc3_1 = nn.Linear(hidden_features, 8)

        # For Lower body clothing
        self.fc4 = nn.Linear(in_features, hidden_features)
        self.fc4_1 = nn.Linear(hidden_features, 7)

        # For Sleeve length
        self.fc5 = nn.Linear(in_features, hidden_features)
        self.fc5_1 = nn.Linear(hidden_features, 3)

        # For Carry item
        self.fc6 = nn.Linear(in_features, hidden_features)
        self.fc6_1_1 = nn.Linear(hidden_features, 1)
        self.fc6_1_2 = nn.Linear(hidden_features, 1)
        self.fc6_1_3 = nn.Linear(hidden_features, 1)
        
        # For Head wear
        self.fc7 = nn.Linear(in_features, hidden_features)
        self.fc7_1 = nn.Linear(hidden_features, 2)

        # For Foot wear
        self.fc8 = nn.Linear(in_features, hidden_features)
        self.fc8_1 = nn.Linear(hidden_features, 4)

        # For Pose
        self.fc9 = nn.Linear(in_features, hidden_features)
        self.fc9_1 = nn.Linear(hidden_features, 3)

        # For View
        self.fc10 = nn.Linear(in_features, hidden_features)
        self.fc10_1 = nn.Linear(hidden_features, 3)
        
    def forward(self, x):

        x = x.view(x.size(0), -1)
        
        out1 = F.relu(self.fc1(x))
        out1 = self.fc1_1(out1)

        out2 = F.relu(self.fc2(x))
        out2 = self.fc2_1(out2)

        out3 = F.relu(self.fc3(x))
        out3 = self.fc3_1(out3)

        out4 = F.relu(self.fc4(x))
        out4 = self.fc4_1(out4)

        out5 = F.relu(self.fc5(x))
        out5 = self.fc5_1(out5)

        out6 = F.relu(self.fc6(x))
        out6 = self.fc6_1_1(out6)

        out7 = F.relu(self.fc6(x))
        out7 = self.fc6_1_2(out7)

        out8 = F.relu(self.fc6(x))
        out8 = self.fc6_1_3(out8)

        out9 = F.relu(self.fc7(x))
        out9 = self.fc7_1(out9)

        out10 = F.relu(self.fc8(x))
        out10 = self.fc8_1(out10)

        out11 = F.relu(self.fc9(x))
        out11 = self.fc9_1(out11)

        out12 = F.relu(self.fc10(x))
        out12 = self.fc10_1(out12)
        
        # Concatenating the outputs
        out = torch.cat([out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12], dim=1)
        return out
