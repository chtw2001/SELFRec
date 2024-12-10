import logging
import os


class Log(object):
    # 모델이름/모델이름 시간
    def __init__(self,module,filename):
        self.logger = logging.getLogger(module)
        self.logger.setLevel(level=logging.INFO)
        if not os.path.exists('./log/'):
            os.makedirs('./log/')
        handler = logging.FileHandler('./log/'+filename+'.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def add(self,text):
        self.logger.info(text)
