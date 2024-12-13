from datetime import datetime, timedelta
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L

from sklearn.preprocessing import StandardScaler

class StockNetDataModule(L.LightningDataModule):
    def __init__(
        self, 
        data_path, 
        context_window=32,
        batch_size=16, 
        min_active_stock=10, 
        a_threshold=0.,
        b_threshold=0.,
        num_workers=2
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.context_window = context_window
        self.num_workers = num_workers
        self.min_active_stock = min_active_stock
        self.a_threshold = a_threshold
        self.b_threshold = b_threshold
        self.datasets = {}

    def setup(self, stage=None):
        embeddings = torch.from_numpy(np.load(f'{self.data_path}/all_embeddings.npy'))
        timeseries = torch.from_numpy(np.load(f'{self.data_path}/all_timeseries.npy'))
        
        s1 = datetime.strptime('2014-01-01', "%Y-%m-%d")
        s2 = datetime.strptime('2015-08-01', "%Y-%m-%d")
        s3 = datetime.strptime('2015-10-01', "%Y-%m-%d")
        t = (s2 - s1).days
        v = (s3 - s2).days

        mu = timeseries[:, :t].mean(dim=(0,1))
        std = timeseries[:, :t].std(dim=(0,1))
        timeseries = (timeseries - mu) / std
        
        self.datasets['train'] = StockNetDataset(
            embeddings=embeddings[:, :t],
            timeseries=timeseries[:, :t],
            context_window=self.context_window,
            start_date='2014-01-01',
            min_active_stock=self.min_active_stock,
            a_threshold=self.a_threshold,
            b_threshold=self.b_threshold
        )
        self.datasets['val'] = StockNetDataset(
            embeddings=embeddings[:, t:t+v],
            timeseries=timeseries[:, t:t+v],
            context_window=self.context_window,
            start_date='2015-08-01',
            min_active_stock=self.min_active_stock,
            a_threshold=self.a_threshold,
            b_threshold=self.b_threshold
        )
        self.datasets['test'] = StockNetDataset(
            embeddings=embeddings[:, t+v:],
            timeseries=timeseries[:, t+v:],
            context_window=self.context_window,
            start_date='2015-10-01',
            min_active_stock=self.min_active_stock,
            a_threshold=self.a_threshold,
            b_threshold=self.b_threshold
        )
        

    def train_dataloader(self):
        return DataLoader(
            self.datasets['train'], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(self.datasets['val'], batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.datasets['test'], batch_size=self.batch_size, num_workers=self.num_workers)




class StockNetDataset(Dataset):
    def __init__(
        self, embeddings, timeseries, context_window, start_date, min_active_stock=1, a_threshold=0., b_threshold=0.
    ):
        super().__init__()
        self.context_window = context_window
        self.a_threshold = a_threshold
        self.b_threshold = b_threshold

        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        all_dates = []
        for i in range(timeseries.shape[1]):
            date = start_date + timedelta(days=i)
            all_dates += [torch.tensor([date.year, date.month, date.day])]
        all_dates = torch.stack(all_dates).unsqueeze(0).repeat(timeseries.shape[0], 1, 1)
        self.timeseries = torch.cat([timeseries, all_dates], dim=2)
        self.ids = torch.arange(self.timeseries.shape[0])
        
        mask = (embeddings.sum(2) != 0) & (timeseries.sum(2) != 0)
        time_mask = (mask.sum(0) > min_active_stock)
        stock_mask = (mask.sum(1) > context_window)
        self.timeseries = self.timeseries[stock_mask, :][:, time_mask]
        self.embeddings = embeddings[stock_mask, :][:, time_mask]
        self.ids = self.ids[stock_mask]


    def __getitem__(self, idx):
        ts = self.timeseries[:, idx:idx + self.context_window]
        emb = self.embeddings[:, idx:idx + self.context_window]
        
        label = self.timeseries[:, idx + self.context_window, 0]/ ts[:, -1, 0]
        label[label >= 1. + self.b_threshold] = 1
        label[label < 1 - self.a_threshold] = 0
        label[(label != 1) & (label != 0)] = 2

        return ts.float(), emb.float(), self.ids.long(), label.long()

    def __len__(self):
        return self.timeseries.shape[1] - self.context_window - 1
