import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = (123.68 / 255.0,116.779 / 255.0,103.939 / 255.0)
std = (1.0,1.0,1.0)
vgg19 = torchvision.models.vgg19(pretrained=True)
vgg19.eval()

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self,x):
        return x

vgg19.classifier = Identity()

class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [0,5,10,19,28]

        self.vgg19 = vgg19.features
    
    def forward(self,x):
        outputs = []
        output = x
        for i,layer in enumerate(self.vgg19):
            output = layer(output)
            if i in self.layers:
                outputs.append(output)
        return outputs

vgg19 = VGG19().to(device)

def gram_matrix(x):
    x = x.view(x.shape[0],-1)
    x = torch.matmul(x,x.T)
    return x

def neural_style_transfer(content,style,content_weight,style_weight,lr,steps):
    content = Image.open(content)
    style = Image.open(style)

    transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean,std)])
    content = transform(content).to(device)
    style = transform(style).to(device)
    
    generated = content.clone()
    generated.requires_grad = True
    generated = generated.to(device)
    
    optimizer = torch.optim.Adam([generated],lr=lr)
    for i in range(steps):
        content_features = vgg19(content)
        style_features = vgg19(style)
        generated_features = vgg19(generated)

        content_loss = torch.tensor(0.0).to(device)
        style_loss = torch.tensor(0.0).to(device)

        mse = nn.MSELoss()
        for c_f,s_f,g_f in zip(content_features,style_features,generated_features):
            c_f,s_f,g_f = c_f.to(device),s_f.to(device),g_f.to(device)

            content_loss += mse(c_f,g_f)

            style_gram = gram_matrix(s_f)
            generated_gram = gram_matrix(g_f)

            style_loss += mse(style_gram,generated_gram)
        
        total_loss = (content_loss*content_weight) + (style_loss*style_weight)
        total_loss.backward()

        optimizer.step()
        optimizer.zero_grad()
    return generated
