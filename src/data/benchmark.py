import os

from data import common
from data import srdata

import numpy as np

import torch
import torch.utils.data as data

class Benchmark(srdata.SRData):
    '''
    父类SRData初始化后，Benchmark就可迭代了，丢入迭代器dataloader.Dataloader
    就可以过呢据设置获取获取相应的数据集。
    '''
    def __init__(self, args, name='', train=True, benchmark=True):
        super(Benchmark, self).__init__(
            args, name=name, train=train, benchmark=True
        )

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'benchmark', 'benchmark', self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        if self.input_large:
            self.dir_lr = os.path.join(self.apath, 'LR_bicubicL')
        else:
            self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = ('png', '.png')

if __name__ == '__main__':
    from option import args

    a = Benchmark(args, name='Set5', train=False)
    print(a[0])