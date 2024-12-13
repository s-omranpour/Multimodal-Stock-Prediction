import os
import torch
import argparse

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from src.data import StockNetDataModule
from src.model import StockPredictor


print(f'Cuda {"IS" if torch.cuda.is_available() else "is NOT"} Available')

def load_data():
    return StockNetDataModule(
        data_path='path-to-data/', 
        context_window=5,
        batch_size=8, 
        min_active_stock=1, 
        a_threshold=0.0002,
        b_threshold=0.0055,
        num_workers=1
    )

def load_model(mode=0, d_hidden=64, n_blocks=4, n_head=1, dropout=0., attention=0):
    model = StockPredictor(
        n_features=6,
        d_emb=768,
        d_hidden=d_hidden,
        max_ids = 87,
        n_blocks = n_blocks, 
        d_head = d_hidden // n_head, 
        n_head = n_head, 
        dropout=dropout, 
        use_linear_att=False if attention == 0 else True,
        rotary_emb_list=[2],
        ignore_list=None,
        mode=mode,
        lr=5e-4,
        weight_decay=0.
    )
    print('Model #params:', model.count_parameters())
    return model

def load_trainer(mode=0, d_hidden=64, n_blocks=4, n_head=1, dropout=0., attention=0):
    name = f'd_hidden={d_hidden}-n_blocks={n_blocks}-n_head={n_head}-dropout={dropout}'
    proj = {
        0: 'price',
        1: 'tweet',
        2: 'multi'
    }[mode] + '-' + {
        0 : 'std',
        1 : 'linear'
    }[attention]
    wandb_logger = WandbLogger(
        name=name,
        project="stocknet-" + proj,
        save_dir='experiments/' + proj
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"experiments/{proj}/weights", 
        filename=name + "-{epoch:02d}-{val_f1:.2f}",
        save_top_k=1, 
        monitor="val_f1", 
        mode='max'
    )
    early_stop_callback = EarlyStopping(
        monitor="val_f1", min_delta=0.005, patience=100, verbose=False, mode="max"
    )
    return L.Trainer(
        max_epochs=1000,
        accelerator="gpu", 
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        accumulate_grad_batches=1,
        gradient_clip_val=5.,
        enable_progress_bar=False
    )


def main():
    parser = argparse.ArgumentParser(
        "stocknet", add_help=False
    )
    parser.add_argument(
        "--mode", default=0, type=int, help="modality of data"
    )
    parser.add_argument(
        "--nhead", default=1, type=int, help="number of heads"
    )
    parser.add_argument(
        "--hidden", default=1, type=int, help="number of hidden dimensions"
    )
    parser.add_argument(
        "--nblocks", default=1, type=int, help="number of blocks"
    )
    parser.add_argument(
        "--dropout", default=0., type=float, help="dropout"
    )
    parser.add_argument(
        "--attention", default=0, type=int, help="attention type, 0 for standard, 1 for linear"
    )
    args = parser.parse_args()
    datamod = load_data()
    model = load_model(args.mode, args.hidden, args.nblocks, args.nhead, args.dropout, args.attention)
    trainer = load_trainer(args.mode, args.hidden, args.nblocks, args.nhead, args.dropout, args.attention)
    trainer.fit(model, datamodule=datamod)
    trainer.test(ckpt_path='best', datamodule=datamod)


if __name__ == '__main__':
    main()


