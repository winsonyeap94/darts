"""
Custom logging using Loguru.
This method is applied over the basic `logging` package in Python as the basic package is not compatible with
`uvicorn`'s logger.
"""


import os
import sys
import json
import logging
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from functools import partial, partialmethod


class InterceptHandler(logging.Handler):
    log_level_mapping = {
        50: 'CRITICAL',
        40: 'ERROR',
        30: 'WARNING',
        20: 'INFO',
        10: 'DEBUG',
        0: 'NOTSET',
    }

    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except AttributeError:
            level = self.log_level_mapping[record.levelno]

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        log = logger.bind(request_id='app')
        log.opt(
            depth=depth,
            exception=record.exc_info
        ).log(level, record.getMessage())


class LoguruLogger:

    LOGGER_CONFIG_PATH = "./base/loguru_config.json"

    @classmethod
    def make_logger(cls):

        config = cls.load_logging_config(cls.LOGGER_CONFIG_PATH)
        logging_config = config.get('logger')

        # Creating log path if doesn't exist
        Path(logging_config.get('path')).mkdir(parents=True, exist_ok=True)

        loguru_logger = cls.customize_logging(
            Path(logging_config.get('path'), logging_config.get('filename')),
            level=logging_config.get('level'),
            retention=logging_config.get('retention'),
            rotation=logging_config.get('rotation'),
            format=logging_config.get('format')
        )

        # Adding custom levels to the logger
        loguru_logger.level('START', no=33, icon='▶️', color='<green>')
        loguru_logger.level('END', no=35, icon='✔️', color='<green>')
        loguru_logger.level('OUTPUT', no=40, icon='✒️', color='<blue>')

        loguru_logger.__class__.start = partialmethod(loguru_logger.__class__.log, 'START')
        loguru_logger.__class__.end = partialmethod(loguru_logger.__class__.log, 'END')
        loguru_logger.__class__.output = partialmethod(loguru_logger.__class__.log, 'OUTPUT')

        return loguru_logger

    @classmethod
    def customize_logging(cls, filepath: Path, level: str, rotation: str, retention: str, format: str):

        # Removing default logger settings
        logger.remove()

        # Disabling specific packages with excessive DEBUG logging
        logger.disable("shap")
        logger.disable("matplotlib")

        # Enable tqdm logging to console alongside loguru
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

        # Setting based on config settings
        logger.add(
            str(filepath),
            rotation=rotation,
            retention=retention,
            enqueue=True,
            backtrace=True,
            level=level.upper(),
            format=format,
            colorize=True
        )

        logging.basicConfig(handlers=[InterceptHandler()], level=0)
        logging.getLogger("uvicorn.access").handlers = [InterceptHandler()]
        for _log in ['uvicorn', 'uvicorn.error', 'fastapi']:
            _logger = logging.getLogger(_log)
            _logger.handlers = [InterceptHandler()]

        return logger.bind(request_id=None, method=None)

    @classmethod
    def load_logging_config(cls, config_path):
        config = None
        with open(config_path) as config_file:
            config = json.load(config_file)
        return config

    