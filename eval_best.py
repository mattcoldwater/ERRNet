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
import os

# python eval_best.py --name my_errnet --hyper -r --unaligned_loss vgg

if __name__ == "__main__":
    mp.set_start_method('spawn')
    
    opt = TrainOptions().parse()

    opt.isTrain = True
    cudnn.benchmark = True
    opt.no_log = True
    opt.display_id=0
    opt.verbose = False

    datadir = opt.root_dir # datadir = '/media/kaixuan/DATA/Papers/Code/Data/Reflection/'

    # ----------------------- data preparation -------------------
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
    best_val_loss = 1e-6
    for root, _, fnames in sorted(os.walk('/data1/kangfu/haoyu/Data/Reflection/checkpoints/my_errnet/')):
        for fname in fnames:
            if fname.endswith('.pt') and 'best' not in fname:
                engine = Engine(opt)
                engine.best_val_loss = best_val_loss
                engine.val_loss_dir = -1
                opt.icnn_path = os.path.join(root, fname)
                # engine.eval(eval_dataloader_real, dataset_name='testdata_real20', loss_key='SSIM')
                engine.eval(eval_dataloader_ceilnet, dataset_name='testdata_table2', loss_key='SSIM')   
                torch.cuda.empty_cache()
                best_val_loss = engine.best_val_loss
                del engine 
    