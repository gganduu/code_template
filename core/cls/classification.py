import torch
from torchvision import datasets, transforms
import torch.utils.data as data
from utils import common
from core import dataset
from pathlib import Path
import random
from PIL import Image
from matplotlib import pyplot as plt
import math
import numpy as np

logger = common.get_logger('logs', name='classification')

def load_data(config):
    '''
    this function is to load the image data for classification
    :param config: a dict parsed from yaml file
    :return: a train and val data loader
    '''
    ds_para = config.get('dataset')
    ds_dir = ds_para.get('dataset_dir')
    batch_size = config.get('train').get('batch_size')
    train_ds = datasets.ImageFolder(root=Path(ds_dir)/'train', transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=ds_para.get('mean'), std=ds_para.get('std'))
    ]))
    train_loader = data.DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_ds = datasets.ImageFolder(root=Path(ds_dir)/'val', transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=ds_para.get('mean'), std=ds_para.get('std'))
    ]))
    val_loader = data.DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader

@common.calc_time
def train(config, device):
    '''
    this function is to train the classification model and save the best model
    :param config: a dict parsed from yaml file
    :param device: specify the hardware
    :return: None
    '''
    bst_acc = 0.0
    bst_loss = 1.0
    epochs = config.get('train').get('epochs')
    batch_size = config.get('train').get('batch_size')
    train_loader, val_loader = load_data(config)
    model = common.create_model(config)
    loss_func = common.parse_loss_func(config)
    optimizer = common.parse_optimizer(config, model)
    lr_scheduler = common.parse_scheduler(config, optimizer)
    logger.info('start training...')
    model = model.to(device)
    for epoch in range(epochs):
        model.train()
        logger.info('-' * 30)
        train_epo_loss = 0.0
        val_epo_loss = 0.0
        for train_X, train_y in train_loader:
            train_X = train_X.to(device)
            train_y = train_y.to(device)
            train_preds = model(train_X)
            loss_train = loss_func(train_preds, train_y)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            step_loss = loss_train.item() * batch_size
            train_epo_loss += step_loss
        lr_scheduler.step()
        num_corr = 0
        with torch.no_grad():
            model.eval()
            for val_X, val_y in val_loader:
                val_X = val_X.to(device)
                val_y = val_y.to(device)
                val_preds = model(val_X)
                val_loss = loss_func(val_preds, val_y)
                val_epo_loss += val_loss.item() * batch_size
                num_corr += torch.sum(torch.argmax(val_preds, dim=1) == val_y)
        val_epo_acc = num_corr.item() / len(val_loader.dataset)
        if val_epo_acc > bst_acc or (val_epo_loss < bst_loss and val_epo_acc == bst_acc):
            torch.save(model, 'models/best_cls.pkl')
            logger.info('the best model has been saved at models folder')
            bst_acc = val_epo_acc
            bst_loss = val_epo_loss
        logger.info('epoch: {}/{}, train loss: {}, val loss: {}, val accuracy: {}'.format(epoch+1, epochs, train_epo_loss, val_epo_loss, val_epo_acc))

def data_postprocessing(root, preds, num_sample=9):
    '''
    this function is to show samples of predition
    :param root: a string path of input image folders
    :param preds: model predicting results
    :param num_sample: num of samples images to show
    :return: None
    '''
    if Path(root).is_dir():
        test_datasets = sorted([p for p in Path(root).iterdir()])
        image_label = random.sample(list(zip(test_datasets, preds)), num_sample)
        nrows = int(math.sqrt(num_sample))
        ncols = math.ceil(num_sample / int(math.sqrt(num_sample)))
        figs, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(90, 90))
        for i in range(nrows):
            for j in range(ncols):
                axes[i, j].imshow(Image.open(image_label[i*ncols+j][0]))
                axes[i, j].set_title(image_label[i*ncols+j][1])
                axes[i, j].axis('off')
    else:
        plt.imshow(Image.open(root))
        plt.title(preds[0])
    plt.show()

@common.calc_time
def test(config, input, device):
    '''
    this function is to do the inference for given input
    :param config: a dict parsed from yaml file
    :param input: a string of path, it could be a fold or an image
    :param device: specify the hardware
    :return: None
    '''
    ds_para = config.get('dataset')
    test_para = config.get('test')
    results = list()
    model = torch.load(test_para.get('model'))
    model.to(device)
    model.eval()
    if Path(input).is_dir():
        test_ds = dataset.ImageDataset(input, transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=ds_para.get('mean'), std=ds_para.get('std'))
        ]))
        test_loader = data.DataLoader(dataset=test_ds, batch_size=test_para.get('batch_size'), shuffle=False, num_workers=0)
        with torch.no_grad():
            for test_X in test_loader:
                test_X = test_X.to(device)
                preds = model(test_X)
                results.extend(torch.argmax(preds, dim=1).cpu().numpy())
    elif Path(input).is_file():
        image = Image.open(input)
        image = common.resize_with_aspect(image, (224, 224))
        image_np = np.array(image, dtype=np.float32) / 255.0
        pred = model(torch.from_numpy(np.expand_dims(image_np, axis=0).transpose((0, 3, 1, 2))).to(device))
        results.append(torch.argmax(pred, dim=1).cpu().numpy())
    return results