# Add to path
import sys
import pandas as pd
import torch
from torch import nn
import numpy as np
from torch import optim
import tqdm
from meta_dmoe import get_dataloader
import hydra
from meta_dmoe import train_model, train_model_selector, train_kd

LOAD_EXPERTS = True
DEVICE = 'cpu'

sys.path.append('../domain-adaptation')



class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, climate):
        self.df = df
        if climate is not None:
            self.X_source_domain = df[df.climate == climate].iloc[:,6:].copy()
            self.y_source_domain = df[df.climate == climate]['fact_temperature'].copy()
            self.climate = climate
        else:
            self.X_source_domain = df.iloc[:,6:].copy()
            self.y_source_domain = df['fact_temperature'].copy()
            self.climate = climate

        assert len(self.X_source_domain) == len(self.y_source_domain)

    def __len__(self):
        return len(self.y_source_domain)

    def __getitem__(self, index):
        X = torch.tensor(self.X_source_domain.iloc[index].values).to(torch.float32)
        y = torch.tensor(self.y_source_domain.iloc[index]).to(torch.float32)
        metadata = {
            'climate': self.climate if self.climate is not None else self.df.iloc[index].climate
        }
        return X, y, metadata

def initializeExperts(cfg):

    source_domains_experts = {}

    for climate in domains_train:
        exp_config = cfg['expert' + 'climate']
        source_domains_experts[climate] = FTTransformer(exp_config=model_config.d_numerical,
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
    
    with hydra.initialize(version_base=None, config_path="../configs"):
        data_cfg = hydra.compose(config_name='data_config')
        experts_cfg = hydra.compose(config_name='experts_config')
        student_cfg = hydra.compose(config_name='student_config')

    df_train = pd.read_csv('/canonical-paritioned-dataset/shifts_canonical_train.csv') #train
    df_dev_in = pd.read_csv('/canonical-paritioned-dataset/shifts_canonical_dev_in.csv')
    df_dev_out = pd.read_csv('/canonical-paritioned-dataset/shifts_canonical_dev_out.csv')

    df_dev = pd.concat([df_dev_in, df_dev_out]) # eval dataset

    domains_train = df_train.climate.unique()

    batch_size = 256
    train_loader = torch.utils.data.DataLoader(Dataset(df_train, None), batch_size = data_cfg.batch_size)
    val_loader = torch.utils.data.DataLoader(Dataset(df_dev, None), batch_size = data_cfg.batch_size)
    
    
    
    
    
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
        expert.load_state_dict(torch.load(f"trained_experts/{climate}.pth"))
    models_list = get_feature_list(source_domains_experts, device=device)

    selector = fa_selector(dim=student_cfg.fa_selector.dim, 
                           depth=student_cfg.fa_selector.depth, 
                           heads=student_cfg.fa_selector.heads, 
                           mlp_dim=student_cfg.fa_selector.mlp_dim, 
                           dropout=student_cfg.fa_selector.dropout,
                           out_dim=student_cfg.fa_selector.out_dim).to(device)
    
    if args.load_pretrained_aggregator:
        print("Skip pretraining knowledge aggregator...")
    else:
        print("Pretraining knowledge aggregator...")
        
        train_model_selector(selector, name+'_pretrained', models_list, device, train_loader, val_loader, root_dir=args.data_dir,
                             num_epochs=args.aggregator_pretrain_epoch, save=True, batch_size=data_cfg.batch_size, lr=1e-4, l2=0,
                             num_epochs=12, decayRate=0.96)

    selector.load_state_dict(torch.load(f"model/{args.dataset}/{name}_pretrained_selector_best.pth"))

    student = StudentModel(model.config.student, device=device, num_classes=OUT_DIM[args.dataset])

    if args.load_pretrained_student:
        print("Skip pretraining student...")
    else:
        print("Pretraining student...")
        train_model(student, name+"_pretrained", device=device, 
                    num_epochs=args.student_pretrain_epoch, save=True,
                    root_dir=args.data_dir)

    student.load_state_dict(torch.load(f"model/{args.dataset}/{name}_pretrained_exp_best.pth"))

    print("Start meta-training...")
    train_kd(selector, name+"_meta", models_list, student, name+"_meta", split_to_cluster,
            device=device, batch_size=args.batch_size, sup_size=args.sup_size, 
            tlr=args.tlr, slr=args.slr, ilr=args.ilr, num_epochs=args.epoch, save=True, test_way='ood',
            root_dir=args.data_dir)





