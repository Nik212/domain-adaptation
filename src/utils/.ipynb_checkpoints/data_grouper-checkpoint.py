import numpy as np

class DataGrouper(torch.utils.data.Dataset):
    def __init__(self, df, domain_list, num_domains, num_groups_per_batch=2, batch_size=128):
        self.df = df
        climate_idx = np.random.choice(num_domains, size=2, replace=False, p=1.0/num_domains)
        self.climates = domain_list[climate_idx]
        self.X_source_domain = df[df.climate == self.climates].iloc[:,6:].copy()
        self.y_source_domain = df[df.climate == self.climates]['fact_temperature'].copy()

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