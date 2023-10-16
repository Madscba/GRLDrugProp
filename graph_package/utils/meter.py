from torchdrug.core import Meter
import wandb

class WandbMeter(Meter): 
    def __init__(self, log_interval=100, silent=False):
        super().__init__(log_interval, silent)
        self.fold_id = 1
    
    def log(self, record):
        super().log(record)
        for k in sorted(record.keys()):
            wandb.log({k: record[k], "epoch": self.epoch_id})

