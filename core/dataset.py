import torch.utils.data as data
import torch
from PIL import Image
from pathlib import Path
from xml.etree import cElementTree as ET
import numpy as np

class ImageDataset(data.Dataset):
    '''
    this class is to provide image dataset for inference usage
    '''
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.images = sorted([p for p in Path(root).iterdir()])

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.images)

class CustomDataset(data.Dataset):
    '''
    this class is customized dataset for object detection training usage
    '''
    def __init__(self, root, folder, label_mapping, transform):
        super().__init__()
        self.root = root
        self.label_mapping = label_mapping
        self.transform = transform
        self.images = sorted([p for p in (Path(root)/'JPEGImages').joinpath(folder).rglob('*.jpg')])
        self.annotations = sorted([p for p in (Path(root)/'Annotations').joinpath(folder).rglob('*.xml')])

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        target = dict()
        boxes = list()
        labels = list()
        annotaion = self.annotations[idx]
        tree = ET.parse(annotaion)
        for o in tree.iterfind('object'):
            labels.append(self.label_mapping.get(o.findtext('name')))
            box = list()
            for b in o.find('bndbox'):
                box.append(int(b.text))
            boxes.append(box)
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['image_id'] = torch.as_tensor(idx, dtype=torch.int64)
        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target

    def __len__(self):
        return len(self.images)

class CustomSegDataset(data.Dataset):
    '''
    this class is customized dataset for segmentation training usage
    '''
    def __init__(self, root, folder, label_mapping, transform):
        super().__init__()
        self.root = root
        self.label_mapping = label_mapping
        self.transform = transform
        self.images = sorted([p for p in (Path(root)/'JPEGImages').joinpath(folder).rglob('*.jpg')])
        self.annotations = sorted([p for p in (Path(root)/'Annotations').joinpath(folder).rglob('*.xml')])
        self.masks = sorted([p for p in (Path(root)/'SegmentationObject').joinpath(folder).rglob('*.png')])

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        mask = Image.open(self.masks[idx])
        target = dict()
        boxes = list()
        labels = list()
        annotaion = self.annotations[idx]
        tree = ET.parse(annotaion)
        for o in tree.iterfind('object'):
            labels.append(self.label_mapping.get(o.findtext('name')))
            box = list()
            for b in o.find('bndbox'):
                box.append(int(b.text))
            boxes.append(box)
        mask = np.array(mask)
        # each color corresponds to a different instance
        # with 0 being background, so remove it
        obj_ids = np.unique(mask)[1:]
        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['image_id'] = torch.as_tensor(idx, dtype=torch.int64)
        target['masks'] = torch.as_tensor(masks, dtype=torch.uint8)
        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target

    def __len__(self):
        return len(self.images)

