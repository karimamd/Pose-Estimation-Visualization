from imports import *

"""
Model:
Pretrained model (e.g, resnet50) features except last 4 blocks
+
4 Convolutional Layers + Dropouts
+
3 Deconvolutional Layers + BatchNorm Layers
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TL_Model(nn.Module):
  
  def __init__(self):
    super(TL_Model, self).__init__()
    
    def features_extractor():
      resnet = models.resnet50(pretrained=True)
      ft_ext = torch.nn.Sequential(*(list(resnet.children())[:-4]))
      
      for param in ft_ext.parameters():
          param.requires_grad = False
      
      return ft_ext
    
    
    self.ft = features_extractor()
    self.cn0 = nn.Conv2d(512, 256, 5, 1, 2)
    self.cn1 = nn.Conv2d(256, 112, 3, 1, 1)
    self.cn2 = nn.Conv2d(112, 56, 1, 1, 0)
    self.cn3 = nn.Conv2d(56, 19, 1, 1, 0)
    self.dc = nn.ConvTranspose2d(19,19,4,2,1)
    self.dp =nn.Dropout2d(p=0.3, inplace=False)
    self.bn = nn.BatchNorm2d(19)

    
  def forward(self, x):
    x = F.relu(self.ft(x))
    for i in range(4):
      x = F.relu(self.__dict__['_modules']['cn'+str(i)](x))
      x = self.dp(x)
    
    x = F.relu(self.dc(x))
    x = self.bn(x)
    x = self.dp(x)
    x = F.relu(self.dc(x))
    x = self.bn(x)
    x = self.dp(x)
    x = F.relu(self.dc(x))
    x = self.bn(x)
        
      
    return x

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = cv2.imread(image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=False)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.cuda()  #assumes that you're using GPU