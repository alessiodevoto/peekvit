import os
from pprint import pprint

######################################################## Logging ##################################################################
class SimpleLogger:
    """
    Simple logger for logging to stdout and to a file.
    """
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        if not os.path.exists(os.path.dirname(log_file_path)):
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        self.log_file = open(log_file_path, 'w')
    
    def log(self, *args, **kwargs):
        pprint(*args, **kwargs)
        pprint(*args, **kwargs, stream=self.log_file)
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()