import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.functional import f1_score, accuracy, matthews_corrcoef
import lightning as L

from .modules import RMSNorm
from .transformer import HighOrderTransformer


class StockPredictor(L.LightningModule):
    def __init__(
        self, 
        n_features,
        d_emb,
        d_hidden,
        max_ids = 2048,
        n_blocks = 2, 
        d_head = 16, 
        n_head = 4, 
        dropout=0., 
        use_linear_att=True,
        feature_map='SMReg',
        rotary_emb_list=None,
        ignore_list=None,
        mode=0,
        lr=1e-4,
        weight_decay=0.
    ):
        super().__init__()
        self.save_hyperparameters()
        self.d_hidden = d_hidden
        self.lr = lr
        self.weight_decay = weight_decay
        self.mode = mode # 1=ts-only, 2=emb-only , 3=multimodal

        self.ts_proj = nn.Linear(n_features, d_hidden)
        self.emb_proj = nn.Linear(d_emb, d_hidden)
        self.id_embedding = nn.Embedding(num_embeddings=max_ids, embedding_dim=d_hidden)
        self.mask_token = nn.Parameter(torch.zeros(d_hidden))

        self.encoder = HighOrderTransformer(
            d_hidden, 
            n_blocks, 
            d_head, 
            n_head, 
            dropout, 
            use_linear_att, 
            feature_map,
            rotary_emb_list, 
            ignore_list
        )
        self.decoder = HighOrderTransformer(
            d_hidden, 
            n_blocks, 
            d_head, 
            n_head, 
            dropout, 
            use_linear_att, 
            feature_map,
            rotary_emb_list, 
            ignore_list
        )        
        self.head = nn.Sequential(RMSNorm(d_hidden), nn.Linear(d_hidden, 2))
        self.dropout = nn.Dropout(p=dropout)
        
        
    def forward_encoder(self, x_emb, x_id):
        id_emb = self.id_embedding(x_id).unsqueeze(2)  # (bs, n, 1, d)
        h_emb = self.dropout(self.emb_proj(x_emb))    # (bs, n, t, d)
        h_emb[h_emb.sum(3) == 0.] = self.mask_token   # replacing zero values with trainable mask token
        h_emb = torch.cat([id_emb, h_emb], dim=2)     # (bs, n, 1 + t, d)     
        return self.encoder(h_emb)

    def forward_decoder(self, x_ts, x_id, emb_hiddens=None):
        id_emb = self.id_embedding(x_id).unsqueeze(2)
        h_ts = self.dropout(self.ts_proj(x_ts))
        h_ts[h_ts.sum(3) == 0.] = self.mask_token  
        h_ts = torch.cat([id_emb, h_ts], dim=2)
        h_ts, _ = self.decoder(h_ts, emb_hiddens)
        return h_ts

    def forward(self, x_ts=None, x_emb=None, x_id=None):
        assert (x_id is not None) and (x_ts is not None or x_emb is not None), "Invalid inputs"
        emb_hiddens = None
        if self.mode > 0:
            h_emb, emb_hiddens = self.forward_encoder(x_emb, x_id)
            if self.mode == 1:
                return self.head(h_emb[:, :, 0, :])
        
        h_ts = self.forward_decoder(x_ts, x_id, emb_hiddens)
        return self.head(h_ts[:, :, 0, :])
    

    def calc_metrics(self, logits, labels):
        logits_ = logits.flatten(end_dim=1)
        labels_ = labels.flatten(end_dim=1)
        mask = (labels_ != 2)
        logits_ = logits_[mask]
        labels_ = labels_[mask]
        preds = logits_.argmax(dim=-1)
        
        loss = F.cross_entropy(logits_, labels_)
        f1 = f1_score(preds, labels_, task='binary', num_classes=2)
        acc = accuracy(preds, labels_, task='binary', num_classes=2)
        mcc = matthews_corrcoef(preds, labels_, task='binary', num_classes=2)
        return loss, f1, acc, mcc
    

    def step(self, batch, mode='train'):
        x_ts, x_emb, x_id, label = batch
        logits = self.forward(x_ts, x_emb, x_id)
        loss, f1, acc, mcc = self.calc_metrics(logits, label)

        self.log(f"{mode}_loss", loss.item())
        self.log(f"{mode}_f1", f1.item())
        self.log(f"{mode}_acc", acc.item())
        self.log(f"{mode}_mcc", mcc.item())
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, mode='train')
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, mode='val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, mode='test')
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
        return [optimizer]#, [lr_scheduler]

    