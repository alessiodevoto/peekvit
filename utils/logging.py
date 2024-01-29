import os
import wandb
from pprint import pprint

class SimpleLogger:
    """
    Simple logger for logging to stdout and to a file.
    """
    def __init__(self, settings, dir, log_every=500):
        self.log_file_path = os.path.join(dir, 'log.txt')
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        self.log_file = open(self.log_file_path, 'a+')
        self.every = log_every
        self.counter = -1
        self.log(settings[0])
        
    
    def log(self,  *args, **kwargs):
        if self.counter in {0,-1}:
            print(*args, **kwargs)
            pprint(*args, **kwargs, stream=self.log_file)
            self.log_file.flush()
        self.counter += 1
        self.counter %= self.every
    
    def close(self):
        self.log_file.close()


class WandbLogger:
    """
    Logger for logging to wandb.
    """
    def __init__(self, wandb_entity, wandb_project, settings, dir=None, wandb_run=None):
        self.entity = wandb_entity
        self.project = wandb_project
        self.wandb_run = wandb_run
        self.config = settings if isinstance(settings, dict) else eval(settings)
        
                
        wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            config=self.config,
            name=wandb_run,
            dir=dir,
        )
    
    def log(self, dict_to_log):
        wandb.log(dict_to_log)
    
    def close(self):
        wandb.finish()