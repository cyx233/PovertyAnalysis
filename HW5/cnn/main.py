from wildDM import *
#from data_loader import *
import argparse
from pl_modules import *
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", '-e', type = int, default = 30)
    parser = wildDM.add_argparse_args(parser)
    parser = baseline_module.add_argparse_args(parser)
    
    args = parser.parse_args()
    print(args.batch_size, args.country_name, args.mode, args.urban_flag)
    dm = wildDM(args)
    
    #dm.setup("blah")

    
#     for batch in dm.train_dataloader():
#         x, y = batch
#         print(x.shape, y.shape)
#         break

    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="vloss",
        mode="min",
        dirpath=args.chkpt_path,
        filename="rural-country"+args.country_name+"-{epoch:02d}-{vacc:.2f}",
    )
    
    model = baseline_module(args)

    trainer = pl.Trainer(
        gpus = 1, accelerator = 'gpu', max_epochs = args.epochs, precision = 16,
        strategy = DDPPlugin(find_unused_parameters = False),
        #default_root_dir = args.chkpt_path,
        callbacks = [checkpoint_callback]
    )

    trainer.fit(model, datamodule = dm)

