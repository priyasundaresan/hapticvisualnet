import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import torchvision.models as models
sys.path.insert(0, '/host/src')

class HapticNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None
    
    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]

class HapticVisualNet(nn.Module):
	def __init__(self, rot_dim=3, channels=3, out_classes=3):
		super(HapticVisualNet, self).__init__()
		self.resnet = models.resnet18(pretrained=True)
		self.resnet.conv1 = nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
		modules = list(self.resnet.children())[:-1]      # delete the last fc layer.
		self.resnet = nn.Sequential(*modules)
		self.resnet_out_dim = 512
		self.haptic_out_dim = 128
		self.haptic_net = HapticNet(26, 256, 3, 128)
		self.linear = nn.Linear(self.resnet_out_dim + self.haptic_out_dim, out_features=out_classes)
	def forward(self, img, force):
		features_img = self.resnet(img)
		features_img = features_img.reshape(features_img.size(0), -1)
		features_force = self.haptic_net(force)
		features = torch.cat((features_img, features_force), dim=1).squeeze()
		features = self.linear(features)
		return features

if __name__ == '__main__':
	model = HapticVisualNet().cuda()
	img_test = torch.rand((1,3,200,200)).cuda()
	force_test = torch.rand((1,1,20)).cuda()
	start = time.time()
	result = model.forward(img_test,force_test)
	end = time.time()
	print(end-start)
