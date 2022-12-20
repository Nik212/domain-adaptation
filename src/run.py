# Add to path
import sys
import pandas as pd
import torch
from torch import nn
import numpy as np
from torch import optim
import tqdm
import hydra
from meta_dmoe import train_model, train_model_selector, train_kd
from expertsTraining.ft_transformer import FTTransformer
from models import get_feature_list, fa_selector, StudentModel
from expertsTraining.dataset import DatasetMetaDMOE

LOAD_EXPERTS = True
device = 'cuda'

sys.path.append('../domain-adaptation')



# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, df, climate):
#         self.df = df
#         if climate is not None:
#             self.X_source_domain = df[df.climate == climate].iloc[:,6:].copy()
#             self.y_source_domain = df[df.climate == climate]['fact_temperature'].copy()
#             self.climate = climate
#         else:
#             self.X_source_domain = df.iloc[:,6:].copy()
#             self.y_source_domain = df['fact_temperature'].copy()
#             self.climate = climate

#         assert len(self.X_source_domain) == len(self.y_source_domain)

#     def __len__(self):
#         return len(self.y_source_domain)

#     def __getitem__(self, index):
#         X = torch.tensor(self.X_source_domain.iloc[index].values).to(torch.float32)
#         y = torch.tensor(self.y_source_domain.iloc[index]).to(torch.float32)
#         metadata = {
#             'climate': self.climate if self.climate is not None else self.df.iloc[index].climate
#         }
#         return X, y, metadata
    
    

def initializeExperts(cfg):

    source_domains_experts = {}

    for climate in domains_train:
        exp_config = cfg['expert_' + '_'.join(climate.split(' '))]
        source_domains_experts[climate] = FTTransformer(d_numerical=exp_config.d_numerical,
                                    categories=exp_config.categories,
                                    token_bias=exp_config.token_bias,
                                    d_token=exp_config.d_token,
                                    n_layers=exp_config.n_layers,
                                    n_heads=exp_config.n_heads,
                                    activation=exp_config.activation,
                                    d_ffn_factor=exp_config.d_ffn_factor,
                                    attention_dropout=exp_config.attention_dropout,
                                    ffn_dropout=exp_config.ffn_dropout,
                                    residual_dropout=exp_config.residual_dropout,
                                    prenormalization=exp_config.prenormalization,
                                    initialization=exp_config.initialization,
                                    kv_compression=exp_config.kv_compression, 
                                    kv_compression_sharing=exp_config.kv_compression_sharing,
                                    d_out=exp_config.d_out
                                )
    return source_domains_experts

if __name__ == '__main__':
    
    with hydra.initialize(version_base=None, config_path="../src/configs"):
        data_cfg = hydra.compose(config_name='data_meta_dmoe_config')
        experts_cfg = hydra.compose(config_name='experts_config')
        student_cfg = hydra.compose(config_name='student_config')

    df_train = pd.read_csv('canonical-paritioned-dataset/shifts_canonical_train.csv') #train
    df_dev_in = pd.read_csv('canonical-paritioned-dataset/shifts_canonical_dev_in.csv')
    df_dev_out = pd.read_csv('canonical-paritioned-dataset/shifts_canonical_dev_out.csv')

    df_dev = pd.concat([df_dev_in, df_dev_out]) # eval dataset

    domains_train = df_train.climate.unique()

    train_loader = torch.utils.data.DataLoader(DatasetMetaDMOE(df_train, None), batch_size = data_cfg.batch_size)
    val_loader = torch.utils.data.DataLoader(DatasetMetaDMOE(df_dev, None), batch_size = data_cfg.batch_size)
    
    
    source_domains_experts = initializeExperts(experts_cfg)

    # load experts?
    #if args.load_trained_experts:
    #    print("Skip training domain specific experts...")
    #else:
    #    print("Training domain specific experts...")
    #    train_exp(models_list, all_split, device, batch_size=args.expert_batch_size,
    #            lr=args.expert_lr, l2=args.expert_l2, num_epochs=args.expert_epoch,
    #            save=True, name=name, root_dir=args.data_dir)

    for domain, expert in source_domains_experts.items():
        expert.load_state_dict(torch.load(f"trained_experts/{'_'.join(domain.split(' '))}.pth"))
    models_list = get_feature_list(source_domains_experts, device=device)

    selector = fa_selector(dim=student_cfg.fa_selector.dim, 
                           depth=student_cfg.fa_selector.depth, 
                           heads=student_cfg.fa_selector.heads, 
                           mlp_dim=student_cfg.fa_selector.mlp_dim, 
                           dropout=student_cfg.fa_selector.dropout,
                           out_dim=student_cfg.fa_selector.out_dim).to(device)
    
    if student_cfg.fa_selector.model_path != '':
        print("Skip pretraining knowledge aggregator...")
    else:
        print("Pretraining knowledge aggregator...")
        
        train_model_selector(selector, models_list, device, train_loader, val_loader, root_dir=data_cfg.root_dir,
                             save=True, batch_size=data_cfg.batch_size, lr=data_cfg.selector.lr, l2=data_cfg.selector.l2,
                             num_epochs=data_cfg.selector.num_epochs, decayRate=data_cfg.selector.decayRate)

    selector.load_state_dict(torch.load(student_cfg.fa_selector.model_path))

    student = StudentModel(student_cfg.student, device=device)

    if student_cfg.student.model_path != '':
        print("Skip pretraining student...")
    else:
        print("Pretraining student...")
        train_model(student, device, train_loader, val_loader, 
                    num_epochs=data_cfg.student.num_epochs, save=True,
                    root_dir=data_cfg.root_dir, lr = data_cfg.student.lr, l2=data_cfg.student.l2, decayRate=data_cfg.student.decayRate)

    student.load_state_dict(torch.load(student_cfg.student.model_path))

    print("Start meta-training...")
    train_kd(selector, models_list, device, train_loader, val_loader, student, batch_size=data_cfg.batch_size,  
            tlr=data_cfg.meta.tlr, slr=data_cfg.meta.slr, ilr=data_cfg.meta.ilr, num_epochs=data_cfg.meta.num_epochs, decayRate=data_cfg.meta.decayRate, save=True, test_way='ood',
            root_dir='src/trained_experts')





