from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from mae import MAE
from mae_data import PretrainDataModule


def main():
    data_dir = "/home/asohn3/baraslab/hla/Data/pep2imgtxt/pretrain"
    log_dir = "/home/asohn3/baraslab/hla/Results/logs"

    dset_module = PretrainDataModule(
        root_dir=data_dir,
        batch_size=8192,
        shuffle_dataset=True,
        num_workers=20,
    )

    model = MAE()
    # model = MAE(enc_width=12)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir)

    trainer = Trainer(
        max_epochs=400,
        accelerator='gpu',
        devices=[0],
        # strategy=DDPStrategy(find_unused_parameters=False),
        logger=tb_logger,
        callbacks=[
            ModelCheckpoint(
                dirpath="/home/asohn3/baraslab/hla/Results/checkpoints",
                filename="{epoch}--{avg_val_loss:.4f}",
                save_weights_only=True,
                mode="min",
                monitor="checkpoint_on",
            ),
            LearningRateMonitor("epoch")
        ],
    )

    trainer.fit(model,
                dset_module)


if __name__ == '__main__':
    main()
