# Models
from torch import nn
from nets.transformer import Transformer
import torch
from expertsTraining.ft_transformer import FTTransformer
import typing as ty
import torch.nn.functional as F


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

    
    
    
class FeaturesModel(nn.Module):
    def __init__(self, original_model):
        super(FeaturesModel, self).__init__()
        self.tokenizer = original_model.tokenizer
        self.layers = original_model.layers
        self._start_residual = original_model._start_residual
        self._end_residual = original_model._end_residual
        self._get_kv_compressions = original_model._get_kv_compressions
        self.activation = original_model.activation
        self.ffn_dropout = original_model.ffn_dropout
        self.training = original_model.training
        self.last_normalization = original_model.last_normalization
        self.last_activation = original_model.last_activation
        
        
    def forward(self, x_num, x_cat):
        x = self.tokenizer(x_num, x_cat)
        
        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx + 1 == len(self.layers)
            layer = ty.cast(ty.Dict[str, nn.Module], layer)

            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                # for the last attention, it is enough to process only [CLS]
                (x_residual[:, :1] if is_last_layer else x_residual),
                x_residual,
                *self._get_kv_compressions(layer),
            )
            if is_last_layer:
                x = x[:, : x_residual.shape[1]]
            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)
        assert x.shape[1] == 1
                
        x = x[:, 0]
        if self.last_normalization is not None:
            x = self.last_normalization(x)
        x = self.last_activation(x)
        return x

    
class HeadModel(nn.Module):
    def __init__(self, original_model):
        super(HeadModel, self).__init__()
        self.head = original_model.head
        
    def forward(self, x):
        x = self.head(x)
        x = x.squeeze(-1)
        return x
    
    
class DivideModel(nn.Module):
    def __init__(self, original_model, layer=-1):
        super(DivideModel, self).__init__()
        self.num_ftrs = original_model.head.in_features
        self.num_class = original_model.head.out_features
        self.features = FeaturesModel(original_model)
        # self.features.add_module("avg_pool", nn.AdaptiveAvgPool2d((1,1)))
        self.head = HeadModel(original_model)
        
    def forward(self, x_num, x_cat):
        x = self.features(x_num, x_cat)
        x = x.view(-1, self.num_ftrs)
        x = self.head(x)
        x = x.squeeze(-1)
        return x

    
    
def StudentModel(config, device):
    model = FTTransformer(d_numerical=config.d_numerical,
                        categories=config.categories,
                        token_bias=config.token_bias,
                        d_token=config.d_token,
                        n_layers=config.n_layers,
                        n_heads=config.n_heads,
                        activation=config.activation,
                        d_ffn_factor=config.d_ffn_factor,
                        attention_dropout=config.attention_dropout,
                        ffn_dropout=config.ffn_dropout,
                        residual_dropout=config.residual_dropout,
                        prenormalization=config.prenormalization,
                        initialization=config.initialization,
                        kv_compression=config.kv_compression, 
                        kv_compression_sharing=config.kv_compression_sharing,
                        d_out=config.d_out
                    )
    model = DivideModel(model)
    model = model.to(device)
    return model


class NetFeature(nn.Module):
    def __init__(self, original_model, layer=-1):
        super(NetFeature, self).__init__()
        self.original_model=FeaturesModel(original_model)
        # self.features = nn.ModuleList(list(original_model.children())[:layer])
        
    def forward(self, x_num, x_cat):
        x = self.original_model(x_num, x_cat)
        x = x.view(x.shape[0], -1)
        return x
    

def get_feature_list(models_list, device, layer=-1):
    feature_list = {}
    for climate, model in models_list.items():
        feature_list[climate] = NetFeature(model).to(device)
    return feature_list


def get_models_list(model, device, num_domains=3, pretrained=False, bb='d121'):
    models_list = []
    for _ in range(num_domains+1):
        model = model.to(device)
        models_list.append(model)
    return models_list
