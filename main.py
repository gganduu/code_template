import utils.common as common
import argparse
import torch

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='a config file which contains all training parameters')
    parser.add_argument('-i', '--ipex', action='store_true', help='this is to specify use IPEX or not')
    parser.add_argument('-t', '--train', action='store_true', help='training or inference')
    parser.add_argument('-d', '--data',  help='input data, could be a image or a image path')
    parser.add_argument('-g', '--gpu',  action='store_true', help='gpu or cpu')
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    config = common.parse_yaml(args.config)
    device = torch.device('cuda') if args.gpu else torch.device('cpu')
    type = config.get('model').get('type')
    if type == 'classification':
        model = __import__('core.cls.classification', fromlist=['classification'])
    elif type == 'object_detection':
        model = __import__('core.dec.detection', fromlist=['detection'])
    else:
        model = __import__('core.seg.segmentation', fromlist=['segmentation'])
    if args.ipex:
        import intel_pytorch_extension as ipex
        if args.train:
            model.train(config, ipex.DEVICE)
        else:
            model.test(config, args.data, ipex.DEVICE)
    else:
        if args.train:
            model.train(config, device)
        else:
            model.test(config, args.data, device)
