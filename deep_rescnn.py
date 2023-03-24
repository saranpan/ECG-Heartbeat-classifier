import torch
import torch.nn as nn
import torch.nn.functional as F

class Deep_ResCNN(nn.Module):
    def __init__(self, num_classes = 1):
        super(Deep_ResCNN, self).__init__()
        # kernel size : (5,), number of channel : 32
        self.conv1 = nn.Conv1d(1, 32, 5)
        
        self.conv2_1 = nn.Conv1d(32, 32, 5, padding=2)
        self.conv2_2 = nn.Conv1d(32, 32, 5, padding=2)
        
        self.conv3_1 = nn.Conv1d(32, 32, 5, padding=2)
        self.conv3_2 = nn.Conv1d(32, 32, 5, padding=2)
        
        self.conv4_1 = nn.Conv1d(32, 32, 5, padding=2)
        self.conv4_2 = nn.Conv1d(32, 32, 5, padding=2)
        
        self.conv5_1 = nn.Conv1d(32, 32, 5, padding=2)
        self.conv5_2 = nn.Conv1d(32, 32, 5, padding=2)
        
        self.fc1 = nn.Linear(32*8, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, num_classes)
        
        self._init_weights()

    def forward(self, x):
        x = self.conv1(x)
        
        x1 = F.relu(self.conv2_1(x))
        x1 = self.conv2_2(x1)
        x = F.relu(x + x1)
        x = F.max_pool1d(x, kernel_size=5, stride=2)
        
        x1 = F.relu(self.conv3_1(x))
        x1 = self.conv3_2(x1)
        x = F.relu(x + x1)
        x = F.max_pool1d(x, kernel_size=5, stride=2)
        
        x1 = F.relu(self.conv4_1(x))
        x1 = self.conv4_2(x1)
        x = F.relu(x + x1)
        x = F.max_pool1d(x, kernel_size=5, stride=2)
        
        x1 = F.relu(self.conv5_1(x))
        x1 = self.conv5_2(x1)
        x = F.relu(x + x1)
        x = F.max_pool1d(x, kernel_size=5, stride=2)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

if __name__ == '__main__':
    x = torch.rand(32,1,187)
    res_model = Deep_ResCNN()
    print(res_model)
    out = res_model(x)
    print(out.shape)