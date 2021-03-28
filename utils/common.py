import yaml
import torch
import torchvision
from functools import wraps
import time
import logging
from core.cls import module
from PIL import Image
import shutil
from pathlib import Path
import numpy as np

def get_logger(root, file_level=logging.INFO, console_level=logging.INFO, name='default'):
    '''
    this function is to get a logger
    :param root: the root path of logs
    :param file_level: file handler log level
    :param console_level: stream handler log level
    :param name: the given name of logger
    :return: a logger
    '''
    logger = logging.getLogger(name)
    logger.setLevel(console_level)
    file_handler = logging.FileHandler(filename=root+'/'+time.strftime('%Y%m%d', time.localtime()), mode='a+', encoding='utf8')
    formatter = logging.Formatter('[%(asctime)s] - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(file_level)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(console_level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

logger = get_logger('logs', name='common')

def calc_time(func):
    '''
    this function is to calculate time of func execution
    :param func: a function needs to be decorated
    :return: a decorator function
    '''
    @wraps(func)
    def decorator(*args, **kwargs):
        start = time.time()
        results = func(*args, **kwargs)
        end = time.time()
        logger.info('{} method takes {} seconds'.format(func.__name__, end-start))
        return results
    return decorator

def parse_yaml(path):
    '''
    this function is to parse yaml config file
    :param path: a string path of yaml file
    :return: a dict of config
    '''
    with open(path) as f:
        contents = f.read()
    return yaml.load(contents, yaml.FullLoader)

def create_model(config):
    '''
    this function is to create a model based on config
    :param config: a dict parsed from yaml file
    :return: a model
    '''
    category = config.get('model').get('type')
    if category == 'classification':
        model = module.CustomModule(config)
    elif category == 'object_detection':
        mo_para = config.get('model')
        base = getattr(torchvision.models, mo_para.get('backbone'))
        backbone = torch.nn.Sequential(*list(base(pretrained=True).children())[:mo_para.get('layers')])
        backbone.out_channels = mo_para.get('out_channels')
        rpn_anchor_generator = torchvision.models.detection.faster_rcnn.AnchorGenerator(sizes=mo_para.get('anchor'),
                                                                                        aspect_ratios=mo_para.get(
                                                                                            'aspect_ratio'))
        roi_align = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
        model = torchvision.models.detection.FasterRCNN(backbone=backbone, num_classes=mo_para.get('num_classes'),
                                                        rpn_anchor_generator=rpn_anchor_generator,
                                                        box_roi_pool=roi_align)
    elif category == 'segmentation':
        mo_para = config.get('model')
        base = getattr(torchvision.models, mo_para.get('backbone'))
        backbone = torch.nn.Sequential(*list(base(pretrained=True).children())[:mo_para.get('layers')])
        backbone.out_channels = mo_para.get('out_channels')
        rpn_anchor_generator = torchvision.models.detection.faster_rcnn.AnchorGenerator(sizes=mo_para.get('anchor'),
                                                                                        aspect_ratios=mo_para.get(
                                                                                            'aspect_ratio'))
        roi_align = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
        mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=14, sampling_ratio=2)
        model = torchvision.models.detection.MaskRCNN(backbone=backbone, num_classes=mo_para.get('num_classes'),
                                                      rpn_anchor_generator=rpn_anchor_generator,
                                                      box_roi_pool=roi_align,
                                                      mask_roi_pool=mask_roi_pooler)
    else:
        raise Exception('Not supported category')
    print(model)
    return model

def parse_optimizer(config, model):
    '''
    this function is to parse optimzer from config file
    :param config: a dict parsed from yaml
    :param model: a model
    :return: a optimizer
    '''
    opt_para = config.get('train').get('optimizer')
    name = opt_para.pop('name')
    opt_func = getattr(torch.optim, name)
    optimizer = opt_func(params=model.parameters(), **opt_para)
    return optimizer

def parse_scheduler(config, optimizer):
    '''
    this function is to parse lr_sheduler from config file
    :param config: a dict parsed from yaml
    :param optimizer: a optimzer
    :return: a lr_scheduler
    '''
    sche_para = config.get('train').get('lr_scheduler')
    name = sche_para.pop('name')
    sche_func = getattr(torch.optim.lr_scheduler, name)
    scheduler = sche_func(optimizer=optimizer, **sche_para)
    return scheduler

def parse_loss_func(config):
    '''
    this function i to parse loss function from config file
    :param config: a dict parsed from yaml
    :return: a loss function
    '''
    name = config.get('train').get('loss_func')
    loss_func = getattr(torch.nn, name)()
    return loss_func

def resize_with_aspect(image, size):
    '''
    this function is to resize the image to target size with keeping aspect ratio
    the shorter edge side will be filled with gray color
    :param image: a source image, PIL Image format
    :param size: target size
    :return: a resized image
    '''
    w_s, h_s = image.size
    w_t, h_t = size
    resize_ratio = min((w_t/w_s, h_t/h_s))
    w_r, h_r = w_s * resize_ratio, h_s * resize_ratio
    image_resize = image.resize((int(w_r), int(h_r)), Image.ANTIALIAS)
    image_new = Image.new('RGB', size=size, color=(128, 128, 128))
    image_new.paste(image_resize, ((w_t-int(w_r))//2, (h_t-int(h_r))//2))
    return image_new

def split_dataset(source, target, ratio, *args):
    '''
    this function is to split the dataset to train/val/test by given ratio
    :param source: source dataset path
    :param target: target dataset path
    :param ratio: a ratio tuple/list of train/val/test
    :param args: folders path, do the same actions on these folders
    :return:
    '''
    if Path(target).exists():
        shutil.rmtree(Path(target))
    if len(ratio) == 3:
        Path(target).joinpath(Path(source).stem).joinpath('val').mkdir(parents=True)
    Path(target).joinpath(Path(source).stem).joinpath('train').mkdir(parents=True)
    Path(target).joinpath(Path(source).stem).joinpath('test').mkdir(parents=True)
    data = [f for f in Path(source).iterdir()]
    np.random.shuffle(data)
    train_set = data[:int(ratio[0]*len(data))]
    [shutil.copyfile(tr, Path(target).joinpath(Path(source).stem)/'train'/tr.name) for tr in train_set]
    if len(ratio) == 2:
        test_set = data[int(ratio[0]*len(data)):]
    else:
        val_set = data[int(ratio[0]*len(data)):int((ratio[0]+ratio[1])*len(data))]
        test_set = data[int((ratio[0]+ratio[1])*len(data)):]
        [shutil.copyfile(v, Path(target).joinpath(Path(source).stem)/'val'/v.name) for v in val_set]
    [shutil.copyfile(te, Path(target).joinpath(Path(source).stem)/'test'/te.name) for te in test_set]
    if args is not None:
        for arg in args:
            if len(ratio) == 3:
                Path(target).joinpath(Path(arg).stem).joinpath('val').mkdir(parents=True)
            Path(target).joinpath(Path(arg).stem).joinpath('train').mkdir(parents=True)
            Path(target).joinpath(Path(arg).stem).joinpath('test').mkdir(parents=True)
            for p in Path(target).joinpath(Path(source).stem).iterdir():
                parent = p.stem
                for f1 in p.iterdir():
                    name = f1.stem
                    for f2 in Path(arg).iterdir():
                        if name == f2.stem:
                            shutil.copyfile(f2, Path(target).joinpath(Path(arg).stem).joinpath(parent)/f2.name)