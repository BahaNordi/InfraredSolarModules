import fire
import os
import sys

from InfraredSolarModules.data.cifar100_dataloader import CIFARDataLoader
from InfraredSolarModules.data.dataloader import SolarDataLoader
from InfraredSolarModules.config.yaml_reader import open_yaml
from InfraredSolarModules.models.model import CNN
from InfraredSolarModules.models.resnet_customized import resnet20, resnet32
from InfraredSolarModules.models.densenet_customized import densenet121
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from InfraredSolarModules.utils.metrics import generate_cm
from InfraredSolarModules.utils.plot_confusion_matrix import plot_confusion_matrix
from torch.utils.tensorboard import SummaryWriter


def redefine_fc_layer(model, number_class):
    if hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, number_class)
    elif hasattr(model, 'linear'):
        model.linear = nn.Linear(model.linear.in_features, number_class)

    return model


def train_pipeline(optimizer, train_loader, model, device, epoch, tb_writer,
                   model_name='resnet'):  # default: model_name='resnet'

    # history of loss values in each epoch
    loss_history = {
        "train": []
    }

    for itr, batch in enumerate(train_loader):
        image, labels = batch
        image, labels = image.to(device), labels.to(device)
        optimizer.zero_grad()
        pred = model(image)
        if "resnet" in model_name.lower() or "dense" in model_name.lower():
            pred = F.log_softmax(pred, -1)
        loss = F.nll_loss(pred, labels)
        loss.backward()
        optimizer.step()
        if itr % 20 == 0:
            print('Train iteration {} had loss {:.6f}'.format(itr, loss))
            # ...log the running loss
            tb_writer.add_scalar('training loss',
                                 loss, epoch * len(train_loader) + itr)

        loss_history["train"].append(loss)
        # tb_writer.close()
    loss_total = torch.mean(torch.Tensor(loss_history["train"]))
    return loss_total


def val_pipeline(val_loader, model, device, epoch, tb_writer, number_class, model_name='resnet'):
    # history of loss values in each epoch
    loss_history = {
        "val": []
    }

    correct = 0
    total = 0
    multiclass_correct = list(0. for i in range(number_class))
    multiclass_total = list(0. for i in range(number_class))
    with torch.no_grad():
        all_pred = torch.tensor([], device=device)
        for itr, batch in enumerate(val_loader):
            if itr == 1:
                break
            image, labels = batch
            image, labels = image.to(device), labels.to(device)
            pred = model(image)
            if "resnet" in model_name.lower() or "dense" in model_name.lower():
                pred = F.log_softmax(pred, -1)
            all_pred = torch.cat(
                (all_pred, pred)
                , dim=0
            )
            batch_size = image.shape[0]
            loss = F.nll_loss(pred, labels)
            loss_history["val"].append(loss)
            correct, total, multiclass_correct, multiclass_total = \
                multi_acc(pred, labels, correct, total, multiclass_correct, multiclass_total, batch_size)
            if itr % 20 == 0:
                tb_writer.add_scalar('val loss',
                                     loss, epoch * len(val_loader) + itr)
        loss_total = torch.mean(torch.Tensor(loss_history["val"]))
        cm = generate_cm(val_loader.dataset.targets, all_pred.cpu())
        for i, class_id in enumerate(val_loader.dataset.classes):
            print('Accuracy of class %s:, %2d %%' % (class_id, 100 * multiclass_correct[i] / multiclass_total[i]))
        accuracy_total = 100 * correct / total
    return loss_total, accuracy_total, cm


def multi_acc(pred, labels, correct, total, multiclass_correct, multiclass_total, batch_size):
    _, predicted = torch.max(pred.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    c = (predicted == labels).squeeze()
    for i in range(batch_size):
        label = labels[i]
        multiclass_correct[label] += c[i].item()
        multiclass_total[label] += 1

    return correct, total, multiclass_correct, multiclass_total


def load_checkpoint(checkpoint, device):
    if str(device) == 'cpu':
        checkpoint_dict = torch.load(checkpoint, map_location=torch.device('cpu'))
    else:
        checkpoint_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    return checkpoint_dict


def train(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print('Running model on: ', device)

    data_loader = SolarDataLoader(config)
    # data_loader = CIFARDataLoader(config)

    train_loader = data_loader.train_loader
    val_loader = data_loader.val_loader
    # print(train_loader.__len__())
    number_class = config['train']['number_class']
    lr = config['train']['optim']['lr']
    weight_decay = config['train']['optim']['weight_decay']
    momentum = config['train']['optim']['momentum']
    num_epochs = config['train']['epoch']
    step_size = config['scheduler']['step_size']
    log_dir = config['log']['log_dir']
    gamma = config['scheduler']['gamma']
    model_name = config['train']['model']['model_name']
    pretrained = config['train']['model']['prtrained']
    experiment_name = config['log']['experiment_name']
    checkpoint = config['checkpoint']['init']

    # defining model
    if model_name.lower() == "cnn":
        model = CNN()
    elif model_name.lower() == "resnet20":
        model = resnet20(pretrained=pretrained)
        if pretrained is not None:
            redefine_fc_layer(model, number_class)

    elif model_name.lower() == "resnet32":
        model = resnet32(pretrained=pretrained)
        if pretrained is not None:
            redefine_fc_layer(model, number_class)
    elif model_name.lower() == "densenet":  # in c
        dense_checkpoint = load_checkpoint(pretrained, device)
        model = densenet121(num_class=12, pretrained=dense_checkpoint)  # 12
        if pretrained is not None:
            redefine_fc_layer(model, number_class)
    else:
        raise RuntimeError("Model is not supported")

    model = model.to(device)

    if checkpoint is not None:
        checkpoint_dict = load_checkpoint(checkpoint, device)
        model.load_state_dict(checkpoint_dict)

    if config['train']['optim']['method'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                              weight_decay=weight_decay)
    # train_pipeline(optimizer, train_loader, model)
    scheduler = MultiStepLR(optimizer, milestones=step_size, gamma=gamma)

    # initialize best loss to a large value
    best_loss = float('inf')
    best_accuracy = 0
    tb_writer = SummaryWriter(os.path.join(log_dir, experiment_name))
    for epoch in range(num_epochs):
        print('================= Started Epoch {}/{} ================='.format(epoch + 1, num_epochs))

        # train model on training dataset
        model.train()
        scheduler.step()
        loss_train = train_pipeline(optimizer, train_loader, model, device, epoch, tb_writer, model_name)
        print('Train Loss = {}'.format(loss_train))

        # validation mode
        model.eval()
        loss_val, accuracy_total, cm = val_pipeline(val_loader, model, device, epoch, tb_writer, number_class,
                                                    model_name)
        plot_confusion_matrix(cm, val_loader.dataset.classes, normalize=False,
                              file_name=os.path.join(log_dir, 'confusion_matrix_epoch{}.png'.format(epoch + 1)))
        print('Val Loss = {}'.format(loss_val))
        print('Accuracy of the network on the test images: %.3f' % (
            accuracy_total))
        if loss_val < best_loss or best_accuracy < accuracy_total:
            best_loss = loss_val
            best_accuracy = accuracy_total
            torch.save(model.state_dict(), os.path.join(log_dir, 'model_epoch{}.pth'.format(epoch + 1)))
            print("Copied best model weights!")
    print('End of training!')


if __name__ == "__main__":
    config = open_yaml(sys.argv[1])
    fire.Fire(train(config))
