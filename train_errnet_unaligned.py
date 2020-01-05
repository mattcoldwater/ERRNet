from os.path import join
from options.errnet.train_options import TrainOptions
from engine import Engine
from data.image_folder import read_fns
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets
import util.util as util
import data
import torch.multiprocessing as mp
import torch

def set_learning_rate(lr):
    for optimizer in engine.model.optimizers:
        util.set_opt_param(optimizer, 'lr', lr)

# python train_errnet_unaligned.py --name my_errnet --hyper --unaligned_loss vgg --save_epoch_freq 10

if __name__ == "__main__":
    mp.set_start_method('spawn')
    
    opt = TrainOptions().parse()
    cudnn.benchmark = True
    opt.display_freq = 10 # copy from train_errnet.py
    datadir = opt.root_dir # datadir = '/media/kaixuan/DATA/Papers/Code/Data/Reflection/'

    # ----------------------- data preparation -------------------
    # datadir_syn = join(datadir, 'VOCdevkit/VOC2012/PNGImages')
    # datadir_real = join(datadir, 'real_train')
    datadir_unaligned = join(datadir, 'unaligned', 'unaligned_train400')

    # train_dataset = datasets.CEILDataset(datadir_syn, read_fns('VOC2012_224_train_png.txt'), size=opt.max_dataset_size)
    # train_dataset_real = datasets.CEILTestDataset(datadir_real, enable_transforms=True)
    train_dataset_unaligned = datasets.CEILTestDataset(datadir_unaligned, enable_transforms=True, flag={'unaligned':True}, size=None)
    # train_dataset_fusion = datasets.FusionDataset([train_dataset, train_dataset_unaligned, train_dataset_real], [0.25,0.5,0.25])

    train_dataloader_fusion = datasets.DataLoader(
        train_dataset_unaligned, batch_size=opt.batchSize, shuffle=not opt.serial_batches,  # train_dataset_fusion
        num_workers=opt.nThreads, pin_memory=True)

    eval_dataset_ceilnet = datasets.CEILTestDataset(join(datadir, 'testdata_CEILNET_table2'))

    eval_dataset_real = datasets.CEILTestDataset(
        join(datadir, 'real20'),
        fns=read_fns('real_test.txt'))

    eval_dataloader_ceilnet = datasets.DataLoader(
        eval_dataset_ceilnet, batch_size=1, shuffle=False,
        num_workers=opt.nThreads, pin_memory=True)

    eval_dataloader_real = datasets.DataLoader(
        eval_dataset_real, batch_size=1, shuffle=False,
        num_workers=opt.nThreads, pin_memory=True)

    # -------------------- engine start ---------------------------
    engine = Engine(opt)

    # ----------------------Main Loop for direct training---------------------------
    engine.model.opt.lambda_gan = 0
    # engine.model.opt.lambda_gan = 0.01
    set_learning_rate(1e-4)
    while engine.epoch < 91: # 60
        if engine.epoch == 20:
            engine.model.opt.lambda_gan = 0.01 # gan loss is added after epoch 20
        if engine.epoch == 30:
            set_learning_rate(5e-5)
        if engine.epoch == 40:
            set_learning_rate(1e-5)
        if engine.epoch == 45:
            set_learning_rate(5e-5)
        if engine.epoch == 50:
            set_learning_rate(1e-5)
        ## supposed to be fine tune period
        if engine.epoch == 65:
            set_learning_rate(5e-5)
        if engine.epoch == 70:
            set_learning_rate(1e-5)

        engine.train(train_dataloader_fusion)
        
        # if engine.epoch % 5 == 0:
        #     engine.eval(eval_dataloader_ceilnet, dataset_name='testdata_table2')    
        #     engine.eval(eval_dataloader_real, dataset_name='testdata_real20')

    # ----------------------Main Loop for fine tune------------------------------
    """
    set_learning_rate(1e-4)
    while engine.epoch < 80:
        if engine.epoch == 65:
            set_learning_rate(5e-5)
        if engine.epoch == 70:
            set_learning_rate(1e-5)
            
        engine.train(train_dataloader_fusion)
    """