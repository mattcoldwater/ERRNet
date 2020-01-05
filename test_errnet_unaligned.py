from os.path import join, basename
from options.errnet.train_options import TrainOptions
from engine import Engine
from data.image_folder import read_fns
from data.transforms import __scale_width
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets
import util.util as util
import os
import torch.multiprocessing as mp

# python test_errnet_unaligned.py --name errnet_060 --hyper -r --icnn_path /data1/kangfu/haoyu/Data/Reflection/checkpoints/errnet_060/errnet_060_00463920.pt --unaligned_loss vgg
# python test_errnet_unaligned.py --name my_errnet --hyper -r --icnn_path /data1/kangfu/haoyu/Data/Reflection/checkpoints/my_errnet/errnet_060_00023880.pt --unaligned_loss vgg
# python test_errnet_unaligned.py --name my_errnet --hyper -r --icnn_path /data1/kangfu/haoyu/Data/Reflection/checkpoints/my_errnet/errnet_best_SSIM_testdata_table2.pt --unaligned_loss vgg

if __name__ == "__main__":
    mp.set_start_method('spawn')

    opt = TrainOptions().parse()

    opt.isTrain = False
    cudnn.benchmark = True
    opt.no_log =True
    opt.display_id=0
    opt.verbose = False

    datadir = opt.root_dir # datadir = '/media/kaixuan/DATA/Papers/Code/Data/Reflection/'

    test_dataset_unaligned150 = datasets.RealDataset(join(datadir, 'unaligned/unaligned_test50/blended'))
    test_dataloader_unaligned150 = datasets.DataLoader(
        test_dataset_unaligned150, batch_size=1, shuffle=False,
        num_workers=opt.nThreads, pin_memory=True)

    engine = Engine(opt)

    """Main Loop"""
    result_dir = join(datadir, 'results')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    res = engine.test(test_dataloader_unaligned150, savedir=join(result_dir, 'unaligned_test50'))