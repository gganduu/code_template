import torch
import torchvision

class CustomModule(torch.nn.Module):
    '''
    this class is a customized classification module, it supports to change the backbone
    '''
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_para = config.get('model')
        base_model = getattr(torchvision.models, model_para.get('name'))
        self.base = torch.nn.Sequential(*list(base_model(pretrained=True).children())[:model_para.get('layers')])
        self.avg = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.output = torch.nn.Linear(in_features=model_para.get('in_features'), out_features=model_para.get('num_classes'))

    def forward(self, x):
        x = self.base(x)
        x = self.avg(x)
        x = x.reshape((-1, self.config.get('model').get('in_features')))
        x = self.output(x)
        return x
