from models import fa_selector
from torch import nn
from nets.transformer import Transformer
import torch
from torch import optim
from utils import utils
import pandas as pd
import numpy as np
from tqdm import tqdm
import learn2learn as l2l
import torch.nn.functional as F

def train_epoch(selector, source_domains_experts, student,
                train_loader, val_loader, epoch, device, tlr=1e-4, slr=1e-4, ilr=1e-3,
                batch_size=256, test_way='id', save=False,
                root_dir='data'):
    
    for expert in source_domains_experts.values():
        expert.eval()
    
    loss = nn.MSELoss()

    
    features = student.features #
    head = student.head         #
    features = l2l.algorithms.MAML(features, lr=ilr)  
    features.to(device)
    head.to(device)
    
    all_params = list(features.parameters()) + list(head.parameters())
    optimizer_s = optim.Adam(all_params, lr=slr)
    optimizer_t = optim.Adam(selector.parameters(), lr=tlr)
    
    i = 0
    
    losses = []
    
    iter_per_epoch = len(train_loader)
        
    for x, y_true, metadata in tqdm(train_loader):
        selector.eval()
        head.eval()
        features.eval()
        
        domain = np.array(metadata['climate'])
        
        sup_size = x[0].shape[0]//2
        x_sup_num, x_sup_cat = x[0][:sup_size], x[1][:sup_size]
        y_sup = y_true[:sup_size]
        x_que_num, x_que_cat = x[0][sup_size:], x[1][sup_size:]
        y_que = y_true[sup_size:]
        domain = domain[:sup_size]

        x_sup_num = x_sup_num.to(device)
        x_sup_cat = x_sup_cat.to(device)
        y_sup = y_sup.to(device)
        x_que_num = x_que_num.to(device)
        x_que_cat = x_que_cat.to(device)
        y_que = y_que.to(device)
        

        with torch.no_grad():
            logits = torch.stack(
                [
                utils.features_mask(expert(x_sup_num, x_sup_cat).detach(), domain, climate)
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
        feat = task_model(x_que_num, x_que_cat)
        out = head(feat)
        with torch.no_grad():
            loss_pre = loss(out.squeeze(), y_que).item()/x_que_num.shape[0]
        ###inner loop
        feat = task_model(x_sup_num, x_sup_cat)
        feat = feat.view_as(t_out)

        inner_loss = F.mse_loss(feat, t_out)
        task_model.adapt(inner_loss)
        ###
        ###outer loop
        x_que = task_model(x_que_num, x_que_cat)
        s_que_out = head(x_que)
        s_que_loss = loss(s_que_out.squeeze(), y_que)
        #t_sup_loss = teacher_ce(t_out, y_sup)
        ###
        
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


def train_kd(selector, models_list, device, train_loader, val_loader, student,  batch_size=256, sup_size=24, tlr=1e-4, slr=1e-4, ilr=1e-5, num_epochs=30,
             decayRate=0.96, save=False, test_way='ood', root_dir='data', accu_best=0):
    
    for epoch in range(num_epochs):
        some_train_loss_value = train_epoch(selector, models_list, student, 
                                train_loader, val_loader, epoch, 
                                device, tlr=tlr, slr=slr, ilr=ilr,
                                batch_size=batch_size, test_way=test_way, save=save,
                                root_dir=root_dir) # need to remove some input variables
        # some_eval_loss_value = eval(selector, val_loader, models_list, student, sup_size, device=device, 
        #             ilr=ilr, test=False, progress=False, uniform_over_groups=False,
        #             root_dir=root_dir)

        ### 
        # print results
        # save model

        tlr = tlr*decayRate
        slr = slr*decayRate
        
        
def train_model_selector(selector, models_list, device, train_loader, test_loader, root_dir='data',
                         batch_size=32, lr=1e-6, l2=0,
                         num_epochs=12, decayRate=0.96, save=True, test_way='ood'):
    for model in models_list.values():
        model.eval()
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(selector.parameters(), lr=lr, weight_decay=l2)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    
    i = 0
    
    losses = []
    mse_best = 0

    tot = len(train_loader)
    
    for epoch in range(num_epochs):
        
        print(f"Epoch:{epoch}|| Total:{tot}")
        
        for x, y_true, metadata in tqdm(train_loader):
            selector.train()
            
            x_num, x_cat = x[0], x[1]
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            y_true = y_true.to(device)
    
            with torch.no_grad():
                features = torch.stack([model(x_num, x_cat).detach() for model in models_list.values()], dim=-1)
                features = features.permute((0,2,1))
            out = selector(features)
            out = out.squeeze()
            
            loss = criterion(out, y_true)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item()/batch_size)
            
            
            if i % (tot//2) == 0 and i != 0:
                losses = np.mean(losses)
                avg_mse = get_selector_accuracy(selector, models_list, val_loader, 
                                                device, progress=False)
                
                print("Iter: {} || Train loss: {:.4f} || Val loss: {:.4f} ".format(i, losses, avg_mse))
                losses = []
                
                if avg_mse < mse_best and save:
                    print("Saving model ...")
                    torch.save(model.state_dict(), f'{root_dir}/selector.pth')
                    mse_best = avg_mse
                
            i += 1
        scheduler.step()
        
        
def get_selector_accuracy(selector, models_list, data_loader, device, progress=True):
    selector.eval()
    loss = 0
    total = 0
    mse_loss = nn.MSELoss()
    if progress:
        data_loader = tqdm(data_loader)
    for x, y_true, metadata in data_loader:
        
        x_num, x_cat = x[0], x[1]
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y_true = y_true.to(device)
        
        with torch.no_grad():
            features = torch.stack([model(x_num, x_cat).detach() for model in models_list.values()], dim=-1)
            features = features.permute((0,2,1))
            out = selector(features)
            out = out.squeeze()
            correct += mse_loss(out, y_true).item()
            total += x.shape[0]
    
    return correct/total


        
def train_model(model, device, train_loader, val_loader, domain=None, batch_size=32, lr=1e-3, l2=1e-2, 
                num_epochs=5, decayRate=1., save=True, test_way='ood', root_dir='data'):

    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    
    losses = []
    mse_best = 0
    i = 0 

    tot = len(train_loader)
    
    for epoch in range(num_epochs):
        
        print(f"Epoch:{epoch} || Total:{tot}")
        
        for x, y_true, metadata in tqdm(train_loader):
            model.train()
    
            x_num, x_cat = x[0], x[1]
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            y_true = y_true.to(device)
            
            pred = model(x_num, x_cat)

            loss = criterion(pred, y_true)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item()/batch_size)
            
            
            if i % (tot//2) == 0 and i != 0:
                losses = np.mean(losses)
                avg_mse = get_model_accuracy(model, val_loader, device=device)

                print("Iter: {} || Train loss: {:.4f} || Val loss: {:.4f}".format(i, losses, avg_mse))
                losses = []

                if avg_loss < mse_best and save:
                    print("Saving model ...")
                    torch.save(model.state_dict(), f'{root_dir}/student.pth')
                    mse_best = avg_mse


            i += 1
        scheduler.step()
    
    
def get_model_accuracy(model, data_loader, device, domain=None):
    model.eval()
    mse_loss = nn.MSELoss()
    loss = 0
    total = 0
    for x, y_true, metadata in iter(data_loader):
        
        x_num, x_cat = x[0], x[1]
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y_true = y_true.to(device)
        
        out = model(x_num, x_cat)
        loss += mse_loss(out, y_true).item()
        total += x.shape[0]
        
    return loss / total