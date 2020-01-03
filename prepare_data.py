from os.path import join
from data.image_folder import read_fns, make_dataset, is_image_file
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader
from PIL import Image
import os
from shutil import copyfile

def prepare_pascal():
    my_transform = transforms.Compose([
        transforms.CenterCrop(224),
        # transforms.ToTensor(),
        # transforms.ToPILImage(),
    ])
    datadir = '/data1/kangfu/haoyu/Data/Reflection/VOCdevkit/VOC2012/JPEGImages/'
    targetdir = '/data1/kangfu/haoyu/Data/Reflection/VOCdevkit/VOC2012/PNGImages/'
    if not os.path.exists(targetdir):
        os.mkdir(targetdir)
    paths = make_dataset(datadir)
    for path in paths:
        jpeg = default_loader(path)
        png = my_transform(jpeg)
        png.save(path.replace('JPEGImages', 'PNGImages')[:-3]+'png')

def prepare_CEILNET():
    datadir = '/data1/kangfu/haoyu/Data/Reflection/testdata_CEILNET_table2_raw/'
    transmission_layer_dir = '/data1/kangfu/haoyu/Data/Reflection/testdata_CEILNET_table2/transmission_layer/'
    blended_dir = '/data1/kangfu/haoyu/Data/Reflection/testdata_CEILNET_table2/blended/'
    try:
        os.mkdir('/data1/kangfu/haoyu/Data/Reflection/testdata_CEILNET_table2/')
    except:
        pass
    try:
        os.mkdir(transmission_layer_dir)
    except:
        pass
    try:
        os.mkdir(blended_dir)
    except:
        pass
    for root, _, fnames in sorted(os.walk(datadir)):
        for fname in fnames:
            if is_image_file(fname) and 'label1' in fname:                
                path = os.path.join(root, fname)
                target_path = os.path.join(transmission_layer_dir, fname.replace('-label1', ''))
                copyfile(path, target_path)
            if is_image_file(fname) and 'input' in fname:                
                path = os.path.join(root, fname)
                target_path = os.path.join(blended_dir, fname.replace('-input', ''))
                copyfile(path, target_path)    

def prepare_sir():
    datadir = '/data1/kangfu/haoyu/Data/Reflection/'
    """
    'sir2_withgt'
    'postcard'
    'solidobject'
    'transmission_layer'
    'blended'
    """

if __name__ == "__main__":
    # prepare_pascal()
    # prepare_CEILNET()
    pass