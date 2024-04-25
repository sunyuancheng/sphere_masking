import logging
import wandb
from torch.utils.tensorboard import SummaryWriter
import torch



class FileLogger:
    def __init__(self, is_master=False, is_rank0=False, output_dir=None, logger_name='training'):
        # only call by master 
        # checked outside the class
        self.output_dir = output_dir
        if is_rank0:
            self.logger_name = logger_name
            self.logger = self.get_logger(output_dir, log_to_file=is_master)
        else:
            self.logger_name = None
            self.logger = NoOp()
        
    def get_logger(self, output_dir, log_to_file):
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')

        if output_dir and log_to_file:
            
            time_formatter = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(message)s')
            debuglog = logging.FileHandler(output_dir+'/debug.log')
            debuglog.setLevel(logging.DEBUG)
            debuglog.setFormatter(time_formatter)
            logger.addHandler(debuglog)

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(logging.DEBUG)
        logger.addHandler(console)
        
        # Reference: https://stackoverflow.com/questions/21127360/python-2-7-log-displayed-twice-when-logging-module-is-used-in-two-python-scri
        logger.propagate = False

        return logger

    def console(self, *args):
        self.logger.debug(*args)

    def event(self, *args):
        self.logger.warn(*args)

    def verbose(self, *args):
        self.logger.info(*args)

    def info(self, *args):
        self.logger.info(*args)


class WandbLogger:
    def __init__(self, is_master=False, is_rank0=False, output_dir=None, args=None):
        self.output_dir = output_dir
        if is_rank0:
            self.args = args
            self.logger = self.get_logger()
        else:
            self.args = None
            self.logger = NoOp()
        
    def get_logger(self):
        wandb.init(
            config=self.args,
            name="_".join(self.args.model_name.split('/')[1:-1]),
            dir=self.output_dir,
            project=self.args.project_name,
        )

    def log(self, update_dict, step: int, split: str = ""):
        assert step is not None
        if split != "":
            new_dict = {}
            for key in update_dict:
                new_dict["{}/{}".format(split, key)] = update_dict[key]
            update_dict = new_dict
        wandb.log(update_dict, step=int(step))
    
    def log_plots(self, plots, caption: str = ""):
        assert isinstance(plots, list)
        plots = [wandb.Image(x, caption=caption) for x in plots]
        wandb.log({"data": plots})
    


class TBLogger:
    def __init__(self, is_master=False, is_rank0=False, output_dir=None):
        self.output_dir = output_dir
        if is_rank0:
            self.writer = self.get_logger(output_dir)
        else:
            self.writer = NoOp()

    def get_logger(self, output_dir):
        import os
        return SummaryWriter(os.path.join(self.output_dir, "tsb"))

    def log(self, update_dict, step: int, split: str = ""):
        assert step is not None
        if split != "":
            new_dict = {}
            for key in update_dict:
                new_dict["{}/{}".format(split, key)] = update_dict[key]
            update_dict = new_dict
        for key in update_dict:
            if torch.is_tensor(update_dict[key]):
                self.writer.add_scalar(key, update_dict[key].item(), step)
            else:
                assert isinstance(update_dict[key], int) or isinstance(
                    update_dict[key], float
                )
                self.writer.add_scalar(key, update_dict[key], step)
    

# no_op method/object that accept every signature
class NoOp:
    def __getattr__(self, *args):
        def no_op(*args, **kwargs): pass
        return no_op