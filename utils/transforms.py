import random
from PIL import Image
from torchvision import transforms

class Compose(object):
    '''
    this class is a transform container
    '''
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomHorizontalFlip(object):
    '''
    this class is to flip image and label in horizontal direction
    '''
    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def __call__(self, image, target):
        ratio = random.random()
        if ratio < self.ratio:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            w, _ = image.size
            boxes = target.get('boxes')
            boxes[..., [0, 2]] = w - boxes[..., [2, 0]]
            target['boxes'] = boxes
        return image, target

class ToTensor(object):
    '''
    this class is to convert image to tensor
    '''
    def __call__(self, image, target):
        # return torch.as_tensor(np.array(image), dtype=torch.float32).permute((2, 0, 1))/255, target
        return transforms.ToTensor()(image), target
