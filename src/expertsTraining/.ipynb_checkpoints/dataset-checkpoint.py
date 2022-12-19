import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, climate):
        df = pd.read_csv(root)
        if climate is not None:
            self.X_source_domain = df[df.climate == climate].iloc[:,6:].copy()
            self.y_source_domain = df[df.climate == climate]['fact_temperature'].copy()
            self.climate = climate
        else:
            self.X_source_domain = df.iloc[:,6:].copy()
            self.y_source_domain = df['fact_temperature'].copy()
            self.climate = climate
            
        categorical_cols = [
                  'cmc_available',
                  'gfs_available',
                  'gfs_soil_temperature_available',
                  'wrf_available'
            ]
        

        self.X_n_source_domain = self.X_source_domain.drop(labels=categorical_cols, axis=1)
        
        self.X_n_source_domain = StandardScaler().fit_transform(self.X_n_source_domain)
        self.X_n_source_domain = pd.DataFrame(self.X_n_source_domain).fillna(-999)
        
        self.X_c_source_domain = self.X_source_domain[categorical_cols]
        assert len(self.X_source_domain) == len(self.y_source_domain)

    def __len__(self):
        return len(self.y_source_domain)

    def __getitem__(self, index):
        X_numeric = torch.tensor( self.X_n_source_domain.iloc[index].values).float()
        X_categ = torch.tensor(self.X_c_source_domain.iloc[index].values).long()
        y = torch.tensor(self.y_source_domain.iloc[index]).float()

        return (X_numeric, X_categ), y
    
    

class DatasetMetaDMOE(torch.utils.data.Dataset):
    def __init__(self, df, climate):
        if climate is not None:
            self.X_source_domain = df[df.climate == climate].iloc[:,6:].copy()
            self.y_source_domain = df[df.climate == climate]['fact_temperature'].copy()
            self.climate = climate
        else:
            self.X_source_domain = df.iloc[:,6:].copy()
            self.y_source_domain = df['fact_temperature'].copy()
            self.climate = climate
            self.df=df
            
        categorical_cols = [
                  'cmc_available',
                  'gfs_available',
                  'gfs_soil_temperature_available',
                  'wrf_available'
            ]
        

        self.X_n_source_domain = self.X_source_domain.drop(labels=categorical_cols, axis=1)
        
        self.X_n_source_domain = StandardScaler().fit_transform(self.X_n_source_domain)
        self.X_n_source_domain = pd.DataFrame(self.X_n_source_domain).fillna(-999)
        
        self.X_c_source_domain = self.X_source_domain[categorical_cols]
        assert len(self.X_source_domain) == len(self.y_source_domain)

    def __len__(self):
        return len(self.y_source_domain)

    def __getitem__(self, index):
        X_numeric = torch.tensor(self.X_n_source_domain.iloc[index].values).float()
        X_categ = torch.tensor(self.X_c_source_domain.iloc[index].values).long()
        y = torch.tensor(self.y_source_domain.iloc[index]).float()
        
        metadata = {
            'climate': self.climate if self.climate is not None else self.df.iloc[index].climate
        }

        return (X_numeric, X_categ), y, metadata