import os
import wandb
from pprint import pprint

class SimpleLogger:
    """
    Simple logger for logging to stdout and to a file.
    """
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        if not os.path.exists(os.path.dirname(log_file_path)):
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        self.log_file = open(log_file_path, 'a+')
    
    def log(self,  *args, **kwargs):
        
        print(*args, **kwargs)
        pprint(*args, **kwargs, stream=self.log_file)
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()


class WandbLogger:
    """
    Logger for logging to wandb.
    """
    def __init__(self, entity, project, config, wandb_run=None, wandb_run_dir=None):
        self.entity = entity
        self.project = project
        self.wandb_run = wandb_run
        self.config = config

        wandb.init(
            entity=entity,
            project=project,
            config=config,
            name=wandb_run,
            dir=wandb_run_dir
        )
    
    def log(self, dict_to_log):
        wandb.log(dict_to_log)
    
    def close(self):
        wandb.finish()