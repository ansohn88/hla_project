from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

COL_LIST = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'aliquot_id']


class GenotypeDataset(Dataset):
    def __init__(self,
                 hla_genotype_file: str,
                 col_names: list = COL_LIST
                 ) -> None:
        super().__init__()
        if hla_genotype_file.endswith(".pkl"):
            self.hla_genotype = pd.read_pickle(hla_genotype_file)
        elif hla_genotype_file.endswith(".tsv"):
            self.hla_genotype = pd.read_csv(hla_genotype_file)

        self.hla_genotype = self.hla_genotype[col_names]
        self.col_names = col_names

        self.X, self.y = self.prepare_inputs()

        # y = self.hla_genotype['aliquot_id']
        # y = [
        #     a[:-9] for a in y.tolist()
        # ]

    def prepare_inputs(self):
        for col in self.col_names:
            self.hla_genotype[col] = LabelEncoder().fit_transform(
                self.hla_genotype[col])

        for col in col_names:
            self.hla_genotype[col] = self.hla_genotype[col].astype('category')

        X = self.hla_genotype[col_names[:-1]]
        y = self.hla_genotype[col_names[-1]]

        embedded_cols = {
            n: len(col.cat.categories) for n, col in X.items()
        }

        embedding_sizes = [
            (num_cats, min(50, (num_cats+1)//2))
            for _, num_cats in embedded_cols.items()
        ]

        X = torch.from_numpy(X.values)
        y = torch.from_numpy(y.values)

        return X, y, embedding_sizes

    def __len__(self):
        return len(self.hla_genotype)

    def __getitem__(self, idx):
        X_row = self.X[idx]
        y_row = self.y[idx]
        return {
            'X': X_row,
            'y': y_row
        }


class GenotypeDM(pl.LightningDataModule):
    def __init__(self,
                 hla_genotype_file: str,
                 batch_size: int,
                 num_workers: int,
                 pin_memory: bool,
                 persistent_workers: bool
                 ) -> None:
        super().__init__()
        self.genotype_file = hla_genotype_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            dataset = GenotypeDataset(
                hla_genotype_file=self.genotype_file
            )

            split = int(np.floor(0.2 * len(dataset)))
            self.train_data, self.val_data = random_split(
                dataset, [int(len(dataset) - split), split]
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,
            drop_last=False
        )

    def get_embedding_sizes(self):
        TODO


class GenotypeEmbedder(pl.LightningModule):
    def __init__(
        self
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Embedding()
        )

    def training_step(self, batch, batch_idx):
        # x, _ = batch
        x = batch
        pred = self.mae(x)
        loss = self.masked_mse_loss(x, pred)
        self.log('train_loss', loss, sync_dist=True)
        return {
            'loss': loss
        }

    def validation_step(self, batch, batch_idx):
        # x, _ = batch
        x = batch
        pred = self.mae(x)
        # print(f'Initial X: {x.shape}')
        # print(f'Pred: {pred.shape}')
        loss = self.masked_mse_loss(x, pred)
        self.log('val_loss', loss, sync_dist=True)
        return {
            'output': pred,
            'loss': loss,
            'checkpoint_on': loss
        }

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack(
            [x["loss"] for x in outputs]
        ).mean()
        self.log("avg_loss", avg_loss)

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack(
            [torch.randn(1, requires_grad=True) for _ in outputs]
        ).mean()
        self.log("avg_val_loss", avg_val_loss)
        self.log("checkpoint_on", avg_val_loss)
        return {
            "val_loss": avg_val_loss
        }

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.95),
            weight_decay=0.05
        )

        schedule = torch.optim.lr_scheduler.OneCycleLR(
            optim,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            cycle_momentum=False,
        )

        return {
            'optimizer': optim,
            'lr_scheduler':
            {
                'scheduler': schedule,
                'interval': 'step'
            }
        }
