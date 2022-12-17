# Add to path
import sys
import pandas as pd
import torch
from torch import nn
import numpy as np
from torch import optim
import tqdm
from meta_dmoe import get_dataloader

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

def initializeExperts():

    # implement some logic to initialize experts

    source_domains_experts = {}
    model = nn.Sequential(nn.Linear(123, 1)).to(DEVICE)

    for climate in domains_train:
        source_domains_experts[climate] = model
    return source_domains_experts

if __name__ == '__main__':

    df_train = pd.read_csv('/canonical-paritioned-dataset/shifts_canonical_train.csv') #train
    df_dev_in = pd.read_csv('/canonical-paritioned-dataset/shifts_canonical_dev_in.csv')
    df_dev_out = pd.read_csv('/canonical-paritioned-dataset/shifts_canonical_dev_out.csv')

    df_dev = pd.concat([df_dev_in, df_dev_out]) # eval dataset

    domains_train = df_train.climate.unique()

    batch_size = 64
    train_loader = torch.utils.data.DataLoader(Dataset(df_train, None), batch_size = batch_size)
    val_loader = torch.utils.data.DataLoader(Dataset(df_dev, None), batch_size = batch_size)

    source_domains_experts = initializeExperts()

    # load experts?
    #if args.load_trained_experts:
    #    print("Skip training domain specific experts...")
    #else:
    #    print("Training domain specific experts...")
    #    train_exp(models_list, all_split, device, batch_size=args.expert_batch_size,
    #            lr=args.expert_lr, l2=args.expert_l2, num_epochs=args.expert_epoch,
    #            save=True, name=name, root_dir=args.data_dir)

    for domain,expert in source_domains_experts:
        expert.load_state_dict(torch.load(f"{}"))
    models_list = get_feature_list(source_domains_experts, device=device)

    selector = fa_selector(dim=, depth=, heads=, 
                        mlp_dim==, dropout=args.aggregator_dropout,
                        out_dim=OUT_DIM[args.dataset]).to(device)
    if args.load_pretrained_aggregator:
        print("Skip pretraining knowledge aggregator...")
    else:
        print("Pretraining knowledge aggregator...")
        train_model_selector(selector, name+'_pretrained', models_list, device, root_dir=args.data_dir,
                            num_epochs=args.aggregator_pretrain_epoch, save=True)

    selector.load_state_dict(torch.load(f"model/{args.dataset}/{name}_pretrained_selector_best.pth"))

    student = StudentModel(device=device, num_classes=OUT_DIM[args.dataset])

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





