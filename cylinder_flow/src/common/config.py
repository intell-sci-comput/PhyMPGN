from argparse import ArgumentParser
import yaml
import pprint
import os
import numpy as np
import torch
import random
import os.path as osp


class Config:
    def __init__(self, yaml_filename):
        self.yaml_filename = yaml_filename
        self.data = self.load_config()
        self.process_config()

    @staticmethod
    def get_parser():
        parser = ArgumentParser()
        parser.add_argument('--file', dest='filename', required=True)
        return parser

    def load_config(self):
        if os.path.exists(self.yaml_filename):
            with open(self.yaml_filename, 'r', encoding='utf-8') as stream:
                content = yaml.load(stream, Loader=yaml.FullLoader)
            return content
        else:
            raise IOError('config file don\'t exist!')

    def set_random_seed(self):
        random.seed(self.data['random_seed'])
        np.random.seed(self.data['random_seed'])
        torch.manual_seed(self.data['random_seed'])
        torch.cuda.manual_seed_all(self.data['random_seed'])

    def config_mkdir(self):
        for key in self.data:
            value = self.data[key]
            if type(value) is str and '/' in value:
                path = value.split('/', 1)[0]
                if len(path) > 0 and not osp.exists(path):
                    os.makedirs(path)

    def update_path(self):
        for key in self.data:
            value = self.data[key]
            if type(value) is str:
                count = value.count('{}')
                format_values = [self.data['experiment_name']] + ['{}'] * count
                self.data[key] = value.format(*format_values)

    def set_dtype(self):
        # set data type
        if self.data['precision'] == 'float64':
            self.data['dtype'] = torch.float64
            torch.set_default_dtype(torch.float64)
        else:
            self.data['dtype'] = torch.float32
            torch.set_default_dtype(torch.float32)

    def process_config(self):
        # set random seed
        # self.set_random_seed()
        # set data type
        self.set_dtype()
        # update path in config
        self.update_path()
        # mkdir
        self.config_mkdir()

    def __getitem__(self, key):
        return self.data[key]

    def __str__(self):
        return str(self.data)


def test():
    from kogger import Logger

    args = Config.get_parser().parse_args()
    config = Config(yaml_filename=args.filename)
    logger = Logger('CONFIG')
    logger.info('Load config successfully!')
    logger.info(pprint.pformat(config.data))


if __name__ == '__main__':
    test()
