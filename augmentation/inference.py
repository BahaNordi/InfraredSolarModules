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
                # if itr == 2:
                #     break
                image, labels = batch
                image, labels = image.to(device), labels.to(device)
                pred = model(image)
                pred = F.log_softmax(pred, -1)
                all_predictions = torch.cat([all_predictions, pred], dim=0)
            predictions_ensemble.append(all_predictions)
    all_predictions = torch.mean(torch.stack(predictions_ensemble), dim=0)
    _, predicted = torch.max(all_predictions.data, 1)
    all_labels = data_loader.test_loader.dataset.targets
    correct = (predicted == torch.tensor(all_labels[:all_predictions.shape[0]])).sum().item()
    c = (predicted == torch.tensor(all_labels[:all_predictions.shape[0]])).squeeze()
    multiclass_correct = list(0. for i in range(12))
    multiclass_total = list(0. for i in range(12))
    for i in range(len(all_labels)):
        label = all_labels[i]
        multiclass_correct[label] += c[i].item()
        multiclass_total[label] += 1
    cm = generate_cm(all_labels[:all_predictions.shape[0]], all_predictions)
    print(cm)
    accuracy_total = 100 * correct / all_predictions.shape[0]
    print('Accuracy of the network on the test images: %.3f' % accuracy_total)
    for i, class_id in enumerate(test_loader.dataset.classes):
        print('Accuracy of class %s:, %2d %%' % (class_id, 100 * multiclass_correct[i] / multiclass_total[i]))


if __name__ == "__main__":
    config = open_yaml(sys.argv[1])
    fire.Fire(inference(config))

