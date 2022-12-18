# Models
from torch import nn
from nets.transformer import Transformer
import torch
class ResNetFeature(nn.Module):
    def __init__(self, original_model, layer=-1):
        super(ResNetFeature, self).__init__()
        self.num_ftrs = original_model.classifier.in_features
        self.features = nn.Sequential(*list(original_model.children())[:layer])
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(-1, self.num_ftrs)
        return x

class fa_selector(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout = 0., out_dim=62, pool='mean'):
        super(fa_selector, self).__init__()
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout=dropout)
        self.pool = pool
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim)
        )

    def forward(self, x):
        x = self.transformer(x)
        if self.pool == 'mean':
            x = x.mean(dim=1)
        elif self.pool == 'max':
            x = x.max(dim=1)
        else:
            raise NotImplementedError
        x = self.mlp(x)
        return x
    
    def get_feat(self, x):
        x = self.transformer(x)
        if self.pool == 'mean':
            x = x.mean(dim=1)
        elif self.pool == 'max':
            x = x.max(dim=1)
        else:
            raise NotImplementedError
        return x

class DivideModel(nn.Module):
    def __init__(self, original_model, layer=-1):
        super(DivideModel, self).__init__()
        self.num_ftrs = original_model.classifier.in_features
        self.num_class = original_model.classifier.out_features
        self.features = nn.Sequential(*list(original_model.children())[:layer])
        self.features.add_module("avg_pool", nn.AdaptiveAvgPool2d((1,1)))
        self.classifier = nn.Sequential(*list(original_model.children())[layer:])
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.num_ftrs)
        x = self.classifier(x)
        x = x.view(-1, self.num_class)
        return x

def StudentModel(model, device, num_classes=62, load_path=None):
    model = model
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    if load_path:
        model.load_state_dict(torch.load(load_path))
    model = DivideModel(model)
    model = model.to(device)
    return model

def get_feature_list(models_list, device):
    feature_list = []
    for model in models_list:
        feature_list.append(ResNetFeature(model).to(device))
    return feature_list

def get_models_list(model, device, num_domains=3, num_classes=62, pretrained=False, bb='d121'):
    models_list = []
    for _ in range(num_domains+1):
        model = model.to(device)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        model = model.to(device)
        models_list.append(model)
    return models_list
