import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)#checkpoint对象

def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:#checkpoint类内部属性ok，值为True
            # Data对象,继承了SRData类，初始化会获取数据集，可以通过DataLoader加载
            loader = data.Data(args)
            # Model对象（初始化会构建出模型）
            _model = model.Model(args, checkpoint)

            param = [param.numel() for param in _model.parameters()]
            print('Parameters of model:', sum(param))

            #Loss对象（test_only为True时返回None）
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            #Trainer对象
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done() #关闭日志文件

if __name__ == '__main__':
    main()
