import fire
from InfraredSolarModules.data.dataloader import SolarDataLoader
from config.yaml_reader import open_yaml
from InfraredSolarModules.model import CNN
import torch
import torch.optim as optim
import torch.nn.functional as F


def train_pipeline(optimizer, train_loader, model):

    loss_dict = {'loss': 0}
    model.train()
    for itr, batch in enumerate(train_loader):
        image, label = batch
        optimizer.zero_grad()
        output = model(image)
        loss = F.nll_loss(output, label)
        loss.backward()
        optimizer.step()
        if itr % 100 == 0:
            print('Train iteration {} had loss {:.6f}'.format(itr, loss))
        loss_dict['loss'] = loss


def train(*config_path):
    config = open_yaml(config_path[0])
    data_loader = SolarDataLoader(config)
    train_loader = data_loader.train_loader
    print(train_loader.__len__())

    # defining model
    model = CNN()
    lr = config['train']['optim']['lr']
    weight_decay = config['train']['optim']['weight_decay']
    momentum = config['train']['optim']['momentum']
    if config['train']['optim']['method'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr,  momentum=momentum,
                              weight_decay=weight_decay)
    train_pipeline(optimizer, train_loader, model)






if __name__ == "__main__":
    fire.Fire(train)





