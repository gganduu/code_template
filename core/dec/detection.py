import torch
import torch.utils.data as data
from torchvision import transforms as T
from utils import common, transforms, detection_utils
from core import dataset
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
import math
import numpy as np
import cv2

logger = common.get_logger('logs', name='object_detection')

def collate_fn(batch):
    '''
    this function is to replace default collate_fn to avoid input image having different shape
    the default collate will stack input image tensors, if image tensor has different shape, an error will come out
    :param batch: CustomDataset __getitem__ method's return
    :return: data and target list
    '''
    return tuple(zip(*batch))

def load_data(config):
    '''
    this function is to load data for object detection
    :param config: a dict parsed from yaml file
    :return: a train and val data loader
    '''
    ds_para = config.get('dataset')
    ds_dir = ds_para.get('dataset_dir')
    label_mapping = ds_para.get('label_mapping')
    batch_size = config.get('train').get('batch_size')
    train_ds = dataset.CustomDataset(ds_dir, 'train', label_mapping, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]))
    train_loader = data.DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_ds = dataset.CustomDataset(ds_dir, 'val', label_mapping, transform=transforms.Compose([
        transforms.ToTensor()
    ]))
    val_loader = data.DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    return train_loader, val_loader

@common.calc_time
def train(config, device):
    '''
    this function is to train the object detection model and save the best model
    :param config: a dict parsed from yaml file
    :param device: specify the hardware
    :return: None
    '''
    bst_loss = 1000
    epochs = config.get('train').get('epochs')
    batch_size = config.get('train').get('batch_size')
    train_loader, val_loader = load_data(config)
    model = common.create_model(config)
    optimizer = common.parse_optimizer(config, model)
    lr_scheduler = common.parse_scheduler(config, optimizer)
    logger.info('start training...')
    model = model.to(device)
    total_map = list()
    for epoch in range(epochs):
        model.train()
        logger.info('-' * 30)
        train_epo_loss = 0.0
        for train_X, train_y in train_loader:
            train_X = [x.to(device) for x in train_X]
            train_y = [{k: v.to(device) for k, v in y.items()} for y in train_y]
            loss_train = model(train_X, train_y)
            loss_train_sum = sum(l for l in loss_train.values())
            optimizer.zero_grad()
            loss_train_sum.backward()
            optimizer.step()
            step_loss = loss_train_sum.item() * batch_size
            train_epo_loss += step_loss
        if train_epo_loss / len(train_loader.dataset) < bst_loss:
            torch.save(model, 'models/best_dec.pkl')
            logger.info('the best model has been saved at models folder')
            bst_loss = train_epo_loss / len(train_loader.dataset)
        lr_scheduler.step()
        with torch.no_grad():
            model.eval()
            for val_X, val_y in val_loader:
                val_X = [x.to(device) for x in val_X]
                val_y = [{k: v.to(device) for k, v in y.items()} for y in val_y]
                val_preds = model(val_X)
                step_map, _ = detection_utils.calc_map(val_preds, val_y)
                total_map.append(step_map)
        logger.info('epoch: {}/{}, train loss: {}, MAP: {}'.format(epoch+1, epochs, train_epo_loss / len(train_loader.dataset), sum(total_map)))

def data_postprocessing(root, num_sample=9):
    '''
    this function is to show samples of predition
    :param root: a string path of predicted image folders
    :param num_sample: num of samples images to show
    :return: None
    '''
    test_datasets = sorted([p for p in Path(root).iterdir()])
    nrows = int(math.sqrt(num_sample))
    ncols = math.ceil(num_sample / int(math.sqrt(num_sample)))
    figs, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(90, 90))
    for i in range(nrows):
        for j in range(ncols):
            axes[i, j].imshow(Image.open(test_datasets[i*ncols+j]))
            axes[i, j].axis('off')
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
    test_para = config.get('test')
    model = torch.load(test_para.get('model'))
    model.to(device)
    model.eval()
    if Path(input).is_dir():
        test_ds = dataset.ImageDataset(input, transform=T.Compose([
            T.ToTensor()
        ]))
        test_loader = data.DataLoader(dataset=test_ds, batch_size=test_para.get('batch_size'), shuffle=False, num_workers=0, collate_fn=lambda x: x)
        with torch.no_grad():
            for i, test_X in enumerate(test_loader):
                test_X = [x.to(device) for x in test_X]
                preds = model(test_X)
                for k, item in enumerate(zip(test_X, preds)):
                    # numpy array needs to be wrapped by UMat then convert to numpy array again
                    image = cv2.UMat(np.array(T.ToPILImage()(item[0]))[..., (2, 1, 0)]).get()
                    predictions = detection_utils.nms(item[1])
                    for label, values in predictions.items():
                        boxes = values.get('boxes')
                        scores = values.get('scores')
                        for j in range(len(scores)):
                            cv2.rectangle(image, (int(boxes[j][0]), int(boxes[j][1])),
                                          (int(boxes[j][2]), int(boxes[j][3])), color=(0, 255, 0), thickness=2)
                            cv2.putText(image, str(label) + ':' + str(scores[j].detach().numpy()),
                                        (int(boxes[j][0]), int(boxes[j][1]) + (-15)), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                                        (0, 0, 255), 1)
                    cv2.imwrite('results/' + str(i) + '_' + str(k) + '.jpg', image)
    elif Path(input).is_file():
        image = Image.open(input)
        image_np = np.array(image, dtype=np.float32) / 255.0
        image = cv2.UMat(image_np[..., (2, 1, 0)]).get()
        pred = model(torch.from_numpy(np.expand_dims(image_np, axis=0).transpose((0, 3, 1, 2))).to(device))
        predictions = detection_utils.nms(pred[0])
        for label, values in predictions.items():
            boxes = values.get('boxes')
            scores = values.get('scores')
            for j in range(len(scores)):
                cv2.rectangle(image, (int(boxes[j][0]), int(boxes[j][1])),
                              (int(boxes[j][2]), int(boxes[j][3])), color=(0, 255, 0), thickness=2)
                cv2.putText(image, str(label) + ':' + str(scores[j].detach().numpy()),
                            (int(boxes[j][0]), int(boxes[j][1]) + (-15)), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                            (0, 0, 255), 1)
        cv2.imshow('', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
