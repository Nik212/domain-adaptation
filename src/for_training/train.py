import os
import torch
import hydra
from ft_transformer import FTTransformer
from datamodule import WeatherDatamodule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import LightningModule, Trainer
from torch.optim import AdamW
import torch.nn.functional as F


class Single_expert(LightningModule):
    def __init__(self, config):
        super().__init__()
        
        model_config = config.model
        self.train_config = config.train
        
        self.model=FTTransformer(   d_numerical=model_config.d_numerical,
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
                
        
        if self.train_config.checkpoint_path != '':
             self.model.load_state_dict(torch.load(self.train_config.checkpoint_path, map_location='cpu'))
        
        if not os.path.exists(f'logs/{self.train_config.experiment_name}/'):
                os.makedirs(f'logs/{self.train_config.experiment_name}/')
        
        
        
    def training_step(self, batch, batch_idx):
        
        x_numeric, x_categ, y = batch
        pred = self.model(x_numeric, x_categ)
        loss = F.mse_loss(pred, y)
        self.log('training_loss', loss.item(), prog_bar=True)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        
        x_numeric, x_categ, y = batch
        pred = self.model(x_numeric, x_categ)
        loss = F.mse_loss(pred, y)
        l1_loss = F.l1_loss(pred, y)
        self.log('l1_loss', l1_loss.item(), prog_bar=True)
        self.log('validation_loss', loss.item(), prog_bar=True)
        
        if self.global_rank == 0 and batch_idx == 0:
            torch.save(self.model.state_dict(), f'logs/{self.train_config.experiment_name}/epoch_{self.current_epoch}.pth')
        
        return loss
    
     
    def configure_optimizers(self):
        opt = AdamW(self.model.parameters(), lr=self.train_config.lr, weight_decay=self.train_config.weight_decay)
        shd = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.train_config.max_iterations, eta_min=1e-8)
        return [opt], [shd]
    
    
    
    
    
if __name__ == "__main__":
    with hydra.initialize(version_base=None, config_path="../configs"):
        data_cfg = hydra.compose(config_name='data_config')
        model_cfg = hydra.compose(config_name='model_config')
    
    datamodule = WeatherDatamodule(data_cfg)    
    model = Single_expert(model_cfg)
    
    with open(f'logs/{model_cfg.train.experiment_name}/args.txt', 'wt') as args_file:
        for k, v in sorted(vars(model_cfg).items()):
            args_file.write('%s: %s\n' % (str(k), str(v)))
                
    
    checkpoint_callback = ModelCheckpoint(
        monitor='validation_loss',
        dirpath= f'logs/{model_cfg.train.experiment_name}/lightning_checkpoints/',
        filename='vctk-{epoch:02d}-{validation_loss:.7f}',
        save_top_k = 5)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    ts_logger = TensorBoardLogger(name=model_cfg.train.experiment_name,save_dir=f'lightning_logs/')
    trainer = Trainer(logger=ts_logger, accelerator='gpu', devices=model_cfg.train.num_gpu, max_epochs=model_cfg.train.max_epochs, callbacks=[checkpoint_callback, lr_monitor])
    trainer.fit(model, datamodule=datamodule)
