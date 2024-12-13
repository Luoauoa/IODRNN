import logging
import os
import sys


def config_logging(log_dir, filename='process.log', level=logging.INFO):
    """

    :param log_dir:str, log file path
    :param filename:
    :param level:
    :return:
    """
    # Set the logging format
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(messages)s")
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
    # Create the directory if not exist
    try:
        os.makedirs(log_dir)
    except OSError:
        pass
    file_handler = logging.FileHandler(os.path.join(log_dir, filename))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level=level)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level=level)
    logging.basicConfig(handlers=[file_handler, console_handler], level=level, datefmt=DATE_FORMAT)


def get_logger(log_dir, name, level=logging.INFO, filename='process.log'):
    """

    :param log_dir:
    :param name:
    :param level:
    :param filename:
    :return:
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # file handler
    file_handler = logging.FileHandler(os.path.join(log_dir, filename))
    file_handler.setFormatter(formatter)
    # console handler
    # console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # console_handler = logging.StreamHandler(sys.stdout)
    # console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    # logger.addHandler(console_handler)
    logger.info('Log Directory: %s' % log_dir)
    return logger















