# Models
from torch import nn
from nets.transformer import Transformer
import torch

class fa_selector(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout = 0., out_dim=1, pool='mean'):
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

    
class DivideModel(nn.Module):#??????????????????
    def __init__(self, original_model, layer=-1):
        super(DivideModel, self).__init__()
        self.num_ftrs = original_model.head.in_features
        self.num_class = original_model.head.out_features
        self.features = nn.Sequential(*list(original_model.children())[:layer])
        self.features.add_module("avg_pool", nn.AdaptiveAvgPool2d((1,1)))
        self.head = nn.Sequential(*list(original_model.children())[layer:])
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.num_ftrs)
        x = self.head(x)
        x = x.view(-1, self.num_class)
        return x

    
    
def StudentModel(config, device, load_path=None):
    model = FTTransformer(d_numerical=model_config.d_numerical,
                                    categories=model_config.categories,
                                    token_bias=model_config.token_bias,
                                    d_token=model_config.d_token,
                                    n_layers=model_config.n_layers,
                                    n_heads=model_config.n_heads,
                                    activation=model_config.activation,
                                    d_ffn_factor=model_config.d_ffn_factor,
                                    attention_dropout=model_config.attention_dropout,
                                    ffn_dropout=model_config.ffn_dropout,
                                    residual_dropout=model_config.residual_dropout,
                                    prenormalization=model_config.prenormalization,
                                    initialization=model_config.initialization,
                                    kv_compression=model_config.kv_compression, 
                                    kv_compression_sharing=model_config.kv_compression_sharing,
                                    d_out=model_config.d_out
                                )
    if load_path:
        model.load_state_dict(torch.load(load_path))
    model = DivideModel(model)
    model = model.to(device)
    return model


class NetFeature(nn.Module):
    def __init__(self, original_model, layer=-1):
        super(NetFeature, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:layer])
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        return x
    

def get_feature_list(models_list, device, layer=-1):
    feature_list = []
    for model in models_list:
        feature_list.append(NetFeature(model, layer).to(device))
    return feature_list


def get_models_list(model, device, num_domains=3, pretrained=False, bb='d121'):
    models_list = []
    for _ in range(num_domains+1):
        model = model.to(device)
        models_list.append(model)
    return models_list
