import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNMini(nn.Module):

    def __init__(self, n_classes):
        super(CNNMini, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(576, 128)
        self.fc2 = nn.Linear(128, n_classes)
    
    def forward(self, x):
        
        x = x.view(x.size(0), 1, x.size(1))

        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        
        x = F.relu(self.conv4(x))
        x = F.max_pool1d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv5(x))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        
        x = F.relu(self.conv6(x))
        x = F.max_pool1d(x, kernel_size=2, stride=2)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        x = F.softmax(x, dim=-1)
        
        return x