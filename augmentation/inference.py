import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
from InfraredSolarModules.config.yaml_reader import open_yaml
from InfraredSolarModules.data.dataloader import SolarTestDataLoader
from InfraredSolarModules.models.model import CNN
from InfraredSolarModules.models.resnet_customized import resnet20, resnet32
from InfraredSolarModules.models.densenet_customized import densenet121
from InfraredSolarModules.utils.metrics import generate_cm

if __name__ == "__main__":
    config = open_yaml(sys.argv[1])
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print('Running model on: ', device)
    model = densenet121()
    model = model.to(device)
    checkpoint = torch.load(config['checkpoint']['init'], map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()
    ensemble_rounds = 7
    predictions_ensemble = []
    with torch.no_grad():
        for ensemble in range(0, ensemble_rounds):
            # all_predictions = torch.FloatTensor()
            all_predictions = torch.tensor([], device=device)
            data_loader = SolarTestDataLoader(config, augmentation_index=ensemble)
            test_loader = data_loader.test_loader
            print("Ensemble round {}/{}".format(ensemble + 1, ensemble_rounds))
            for itr, batch in enumerate(data_loader.test_loader):
                # if itr == 3:
                #     break
                image, labels = batch
                image, labels = image.to(device), labels.to(device)
                pred = model(image)
                pred = F.log_softmax(pred, -1)
                all_predictions = torch.cat([all_predictions, pred], dim=0)
            # all_predictions_numpy = F.log_softmax(all_predictions, -1)[:, 1]
            predictions_ensemble.append(all_predictions)
    all_predictions = torch.mean(torch.stack(predictions_ensemble), dim=0)
    cm = generate_cm(data_loader.test_loader.dataset.targets[:all_predictions.shape[0]], all_predictions)
    print(cm)




