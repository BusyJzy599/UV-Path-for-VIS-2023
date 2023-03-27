import logging
import os

FORMATTER = '%(asctime)s - [%(levelname)s]:%(message)s'
# BASE_LOG_PATH = './FlaskServer/logs/'

class MyLogger(object):
    _Levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
    }

    def __init__(self,
                 logger_name: str = "default",
                 file_name: str = "test",
                 level='DEBUG',
                 config=None):
        super().__init__()
        self.logger_file_path = os.path.join(config['logger_path'],file_name+".log") 

        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(MyLogger._Levels[level])

        self.file_handler = logging.FileHandler(self.logger_file_path,'w')
        self.file_handler.setLevel(MyLogger._Levels[level])

        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(MyLogger._Levels[level])

        self.formatter = logging.Formatter(FORMATTER)
        self.file_handler.setFormatter(self.formatter)
        self.console_handler.setFormatter(self.formatter)

        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.console_handler)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)
