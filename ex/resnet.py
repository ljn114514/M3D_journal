import torch.nn as nn
import math, torch
import torch.utils.model_zoo as model_zoo
from torch.nn import init

class MultiConv(nn.Module):
	def __init__(self, planes, stride=1):
		
		super(MultiConv, self).__init__()

		print('M3D layers')
		self.relu = nn.ReLU(inplace=True)

		self.conv1 = nn.Conv3d(planes, planes, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False)
		self.bn1 = nn.BatchNorm3d(planes)
		self.conv1.weight.data.fill_(0)

		self.conv2 = nn.Conv3d(planes, planes, kernel_size=(3, 1, 1), stride=1, dilation=(2, 1, 1), padding=(2, 0, 0), bias=False)
		self.bn2 = nn.BatchNorm3d(planes)
		self.conv2.weight.data.fill_(0)

		self.conv3 = nn.Conv3d(planes, planes, kernel_size=(3, 1, 1), stride=1, dilation=(3, 1, 1), padding=(3, 0, 0), bias=False)
		self.bn3 = nn.BatchNorm3d(planes)
		self.conv3.weight.data.fill_(0)

	def forward(self, x):

		x1 = self.conv1(x)
		x1 = self.bn1(x1)
		x1 = self.relu(x1)

		x2 = self.conv2(x)
		x2 = self.bn2(x2)
		x2 = self.relu(x2)
		x1 = x1 + x2

		x3 = self.conv3(x)
		x3 = self.bn3(x3)
		x3 = self.relu(x3)
		x1 = x1 + x3

		return x1

class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None, M3D = False):
		super(Bottleneck, self).__init__()

		self.M3D = M3D
		self.frames = 16

		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * 4)

		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

		if self.M3D:
			self.conv2_t = MultiConv(planes=planes)

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		if self.M3D:

			#out = out.view(out.size(0)/self.frames, self.frames, out.size(1), out.size(2),out.size(3))
			out = out.unsqueeze(dim=0)
			out = torch.transpose(out, 1, 2)

			out_t = self.conv2_t(out)
			out = out + out_t

			out = torch.transpose(out, 1, 2).contiguous()
			out = out.view(out.size(0)*out.size(1), out.size(2), out.size(3),out.size(4))

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class ResNet(nn.Module):

	def __init__(self, block, layers, num_classes=1000, train=True):
		self.inplanes = 64
		super(ResNet, self).__init__()
		self.istrain = train
		self.frames = 16

		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.conv1_t = MultiConv(planes=64)	

		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.avgpool = nn.AvgPool2d((8,4), stride=1)

		self.num_features = 128
		self.feat = nn.Linear(512 * block.expansion, self.num_features)

		self.feat_bn = nn.BatchNorm1d(self.num_features)
		self.feat_bn1 = nn.BatchNorm1d(self.num_features)
		self.feat_bn2 = nn.BatchNorm1d(self.num_features)
		self.feat_bn3 = nn.BatchNorm1d(self.num_features)
		self.drop = nn.Dropout(0.5)
		self.drop1 = nn.Dropout(0.5)
		self.drop2 = nn.Dropout(0.5)
		self.drop3 = nn.Dropout(0.5)
		self.classifier = nn.Linear(4*self.num_features, num_classes)


		self.feat1 = nn.Conv2d(1, 128, kernel_size=(3,128), stride=1, dilation=(1,1), padding=(1,0), bias=False)
		self.feat2 = nn.Conv2d(1, 128, kernel_size=(3,128), stride=1, dilation=(2,1), padding=(2,0), bias=False)
		self.feat3 = nn.Conv2d(1, 128, kernel_size=(3,128), stride=1, dilation=(3,1), padding=(3,0), bias=False)


	def _make_layer(self, block, planes, blocks, stride=1, frames = 8):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, M3D = True))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		#x_t = x.view(x.size(0)/16, 16, x.size(1), x.size(2),x.size(3))
		x_t = x.unsqueeze(dim=0)
		x_t = torch.transpose(x_t, 1, 2)
		x_t = self.conv1_t(x_t)
		x_t = torch.transpose(x_t, 1, 2).contiguous()
		x_t = x_t.view(-1, x_t.size(2), x_t.size(3), x_t.size(4))
		x = x + x_t

		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.feat(x)

		x = x.unsqueeze(dim=0)
		x0 = x.mean(dim=1)
		x = x.unsqueeze(dim=1)

		x1 = self.feat1(x).mean(dim=2).squeeze(dim=2)
		x2 = self.feat2(x).mean(dim=2).squeeze(dim=2)
		x3 = self.feat3(x).mean(dim=2).squeeze(dim=2)	

		x = torch.cat((x0, x1, x2, x3), dim=1)
		
		return x


def resnet50(pretrained='True', num_classes=1000, train=True):
	model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, train)
	weight = torch.load(pretrained)
	static = model.state_dict()

	for k in static:
		key = 'module.'+k
		if key in weight:
			print('load data',k)
			try:
				static[k].copy_(weight[key])
			except :
				print('*'*100)
				print('error %s'%k)
				print (static[k].size(), weight[k].size())
		else:
			print('not in pretrained model', k)
	return model