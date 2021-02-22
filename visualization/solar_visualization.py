import fire
import sys
import torch
import torch.nn.functional as F
from InfraredSolarModules.config.yaml_reader import open_yaml
from InfraredSolarModules.data.dataloader import SolarTestDataLoader
from InfraredSolarModules.models.model import CNN
from InfraredSolarModules.models.resnet_customized import resnet20, resnet32
from InfraredSolarModules.models.densenet_customized import densenet121
from InfraredSolarModules.utils.metrics import generate_cm
import matplotlib.pyplot as plt


def inference(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print('Running model on: ', device)
    model = densenet121()
    model = model.to(device)
    if str(device) == 'cpu':
        checkpoint = torch.load(config['checkpoint']['init'], map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(config['checkpoint']['init'], map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        data_loader = SolarTestDataLoader(config)
        test_loader = data_loader.test_loader
        x, _ = data_loader.test_loader.sampler.data_source[0]
        plt.imshow(x.numpy()[0], cmap='gray')


if __name__ == "__main__":
    config = open_yaml(sys.argv[1])
    fire.Fire(inference(config))
