import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
import pdb 
from resnet import *

class baseline_module(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model = ResNet18(num_classes = 2, num_channels = 8)
        self.lr = args.lr
        self.t_steps = int(0.9 * 9797) // args.batch_size
        self.oc = args.one_cycle
        self.loss = nn.CrossEntropyLoss() #weights = self.weights)
        
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("resnet")
        parser.add_argument("--lr", '-lr', type = float, default = 0.001)
        parser.add_argument("--one_cycle", '-oc', action = 'store_true')

        return parent_parser

    def forward(self, x):
        out = self.model(x)

        return out

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr = self.lr, momentum = 0.9
        )
        return {'optimizer': optimizer}

#         if self.oc:
#             scheduler = {
#                 'scheduler': torch.optim.lr_scheduler.OneCycleLR(
#                     optimizer, cycle_momentum = True, three_phase = True,
#                     max_lr = self.lr, pcr_start = 0.45, epochs = self.trainer.max_epochs,
#                     steps_per_epoch = self.t_steps
#                 ),
#                 'interval': 'step'
#             }

#             return {'optimizer': optimizer, 'scheduler': scheduler}
#         else:
#             return {'optimizer': optimizer}

    def single_step(self, batch, batch_idx):

        x, y = batch
        y = y.long()

        y_hat_scores = self(x)
        _, y_hat = torch.max(y_hat_scores, 1)

        loss = self.loss(y_hat_scores, y)
        acc = accuracy(y_hat, y)

        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.single_step(batch, batch_idx)

        self.log(
            'tloss', loss, on_epoch = True, on_step = False, logger = True, prog_bar = True
        )

        self.log(
            'tacc', acc, on_epoch = True, on_step = False, logger = True, prog_bar = True
        )

        return loss
        
    def validation_step(self, batch, batch_idx):
        loss, acc = self.single_step(batch, batch_idx)

        self.log(
            'vloss', loss, on_epoch = True, on_step = False, logger = True, prog_bar = True
        )

        self.log(
            'vacc', acc, on_epoch = True, on_step = False, logger = True, prog_bar = True
        )

        return loss
 
    def test_step(self, batch, batch_idx):
        loss, acc = self.single_step(batch, batch_idx)

        self.log(
            'test_loss', loss, on_epoch = True, on_step = False, logger = True, prog_bar = True
        )

        self.log(
            'test_acc', acc, on_epoch = True, on_step = False, logger = True, prog_bar = True
        )

        return loss
 


