from aggregator import fa_selector
from torch import nn
from transformer import Transformer
import torch
from torch import optim

import pandas as pd


df_train = pd.read_csv('/Users/nikglukhov/n.glukhov/canonical-paritioned-dataset/shifts_canonical_train.csv')
df_dev_in = pd.read_csv('/Users/nikglukhov/n.glukhov/canonical-paritioned-dataset/shifts_canonical_dev_in.csv')
df_dev_out = pd.read_csv('/Users/nikglukhov/n.glukhov/canonical-paritioned-dataset/shifts_canonical_dev_out.csv')
df_dev = pd.concat([df_dev_in, df_dev_out])

domains_train = df_train.climate.unique()

def l2_loss(input, target):
    loss = torch.square(target - input)
    loss = torch.mean(loss)
    return loss


def features_mask(features, domains, climate):
    mask = (domains == climate).nonzero()
    features[(domains == climate).nonzero()[0]] = torch.zeros_like(features[0])
    return features


def train_epoch(selector, selector_name, source_domains_experts, student, student_name, 
                train_loader, grouper, epoch, curr, mask_grouper, split_to_cluster,
                device, acc_best=0, tlr=1e-4, slr=1e-4, ilr=1e-3,
                batch_size=256, sup_size=24, test_way='id', save=False,
                root_dir='data'):
    for _, expert in source_domains_experts:
        expert.eval()
    
    student_ce = nn.BCEWithLogitsLoss()

    
    features = student.features
    head = student.classifier
    features.to(device)
    head.to(device)
    
    all_params = list(features.parameters()) + list(head.parameters())
    optimizer_s = optim.Adam(all_params, lr=slr)
    optimizer_t = optim.Adam(selector.parameters(), lr=tlr)
    
    i = 0
    
    losses = []
    
    iter_per_epoch = len(train_loader)
        
    for x, y_true, metadata in train_loader:
        selector.eval()
        head.eval()
        features.eval()
        
        domain = np.array(metadata['climate'])
        
    
        sup_size = x.shape[0]//2
        x_sup = x[:sup_size]
        y_sup = y_true[:sup_size]
        x_que = x[sup_size:]
        y_que = y_true[sup_size:]
        domain = domain[:sup_size]

        x_sup = x_sup.to(device)
        y_sup = y_sup.to(device)
        x_que = x_que.to(device)
        y_que = y_que.to(device)
        

        _squeeze = True
        with torch.no_grad():
            logits = torch.stack(
                [
                features_mask(expert(x_sup).detach(), domain, climate)
                for climate, expert in source_domains_experts.items()
                ], dim=-1)
            ### Expert input: [BS, 123]; Expert output: [BS, N]
            ### logits -> [BS, N, 3].
            logits = logits.permute((0,2,1))
                
            
            #logits = torch.stack([expert(x_sup).detach() for expert in experts_list], dim=-1)
            #logits[:, :, split_to_cluster[z]] = torch.zeros_like(logits[:, :, split_to_cluster[z]])
            #
            #logits = mask_feat(logits, mask, len(models_list), exclude=True)
        
        t_out = selector.get_feat(logits)  

        task_model = features.clone()
        task_model.module.eval()
        feat = task_model(x_que)
        feat = feat.view(feat.shape[0], -1)
        out = head(feat)
        with torch.no_grad():
            loss_pre = student_ce(out, y_que.unsqueeze(-1).float()).item()/x_que.shape[0]
        
        feat = task_model(x_sup)
        feat = feat.view_as(t_out)

        inner_loss = l2_loss(feat, t_out)
        task_model.adapt(inner_loss)
        
        x_que = task_model(x_que)
        x_que = x_que.view(x_que.shape[0], -1)
        s_que_out = head(x_que)
        s_que_loss = student_ce(s_que_out, y_que.unsqueeze(-1).float())
        #t_sup_loss = teacher_ce(t_out, y_sup)
        
        s_que_loss.backward()
        
        optimizer_s.step()
        optimizer_t.step()
        optimizer_s.zero_grad()
        optimizer_t.zero_grad()
        
        ### Print some validation info
        ### Code here
        ###

        losses.append(s_que_loss.item()/x_que.shape[0])
        
            
        i += 1
    return None


def train_kd(selector, train_loader, val_loader, selector_name, models_list, student, student_name, split_to_cluster, device,
             batch_size=256, sup_size=24, tlr=1e-4, slr=1e-4, ilr=1e-5, num_epochs=30,
             decayRate=0.96, save=False, test_way='ood', root_dir='data'):
    
    for epoch in range(num_epochs):
        some_train_loss_value = train_epoch(selector, selector_name, models_list, student, student_name, 
                                train_loader, grouper, epoch, curr, mask_grouper, split_to_cluster,
                                device, acc_best=accu_best, tlr=tlr, slr=slr, ilr=ilr,
                                batch_size=batch_size, sup_size=sup_size, test_way=test_way, save=save,
                                root_dir=root_dir) # need to remove some input variables
        some_eval_loss_value = eval(selector, val_loader, models_list, student, sup_size, device=device, 
                    ilr=ilr, test=False, progress=False, uniform_over_groups=False,
                    root_dir=root_dir)

        ### 
        # print results
        # save model

        tlr = tlr*decayRate
        slr = slr*decayRate