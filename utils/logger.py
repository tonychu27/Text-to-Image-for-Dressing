import datetime
import logging
import time


class MessageLogger():

    def __init__(self, opt, start_iter=1, tb_logger=None):
        self.exp_name = opt['name']
        self.interval = opt['print_freq']
        self.start_iter = start_iter
        self.max_iters = opt['max_iters']
        self.use_tb_logger = opt['use_tb_logger']
        self.tb_logger = tb_logger
        self.start_time = time.time()
        self.logger = get_root_logger()

    def __call__(self, log_vars):
        epoch = log_vars.pop('epoch')
        current_iter = log_vars.pop('iter')
        lrs = log_vars.pop('lrs')

        message = (f'[{self.exp_name[:5]}..][epoch:{epoch:3d}, '
                   f'iter:{current_iter:8,d}, lr:(')
        for v in lrs:
            message += f'{v:.3e},'
        message += ')] '

        if 'time' in log_vars.keys():
            iter_time = log_vars.pop('time')
            data_time = log_vars.pop('data_time')

            total_time = time.time() - self.start_time
            time_sec_avg = total_time / (current_iter - self.start_iter + 1)
            eta_sec = time_sec_avg * (self.max_iters - current_iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            message += f'[eta: {eta_str}, '
            message += f'time: {iter_time:.3f}, data_time: {data_time:.3f}] '

        for k, v in log_vars.items():
            message += f'{k}: {v:.4e} '
            if self.use_tb_logger and 'debug' not in self.exp_name:
                self.tb_logger.add_scalar(k, v, current_iter)

        self.logger.info(message)


def init_tb_logger(log_dir):
    from torch.utils.tensorboard import SummaryWriter
    tb_logger = SummaryWriter(log_dir=log_dir)
    return tb_logger


def get_root_logger(logger_name='base', log_level=logging.INFO, log_file=None):
    logger = logging.getLogger(logger_name)
    if logger.hasHandlers():
        return logger

    format_str = '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s'
    logging.basicConfig(format=format_str, level=log_level)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger
