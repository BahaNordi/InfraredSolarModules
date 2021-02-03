import fire
import os
import sys
from InfraredSolarModules.data.dataloader import SolarDataLoader
from InfraredSolarModules.config.yaml_reader import open_yaml
from InfraredSolarModules.model import CNN
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR


def train_pipeline(optimizer, train_loader, model):

    # history of loss values in each epoch
    loss_history = {
        "train": []
    }

    # histroy of metric values in each epoch
    metric_history = {
        "train": []
    }
    for itr, batch in enumerate(train_loader):
        image, labels = batch
        optimizer.zero_grad()
        output = model(image)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
        if itr % 20 == 0:
            print('Train iteration {} had loss {:.6f}'.format(itr, loss))

        loss_history["train"].append(loss)
    loss_total = torch.mean(torch.Tensor(loss_history["train"]))
    return loss_total


def val_pipeline(val_loader, model):
    # history of loss values in each epoch
    loss_history = {
        "val": []
    }

    # histroy of metric values in each epoch
    metric_history = {
        "val": []
    }
    correct = 0
    total = 0
    multiclass_correct = list(0. for i in range(12))
    multiclass_total = list(0. for i in range(12))
    with torch.no_grad():
        for itr, batch in enumerate(val_loader):
            image, labels = batch
            output = model(image)
            batch_size = image.shape[0]
            loss = F.nll_loss(output, labels)
            loss_history["val"].append(loss)
            correct, total, multiclass_correct, multiclass_total = \
                multi_acc(output, labels, correct, total, multiclass_correct, multiclass_total, batch_size)
        loss_total = torch.mean(torch.Tensor(loss_history["val"]))
        for i, class_id in enumerate(val_loader.dataset.classes):
            print('Accuracy of class %s:, %2d %%' % (class_id, 100 * multiclass_correct[i] / multiclass_total[i]))
        accuracy_total = 100 * correct / total
    return loss_total, accuracy_total


def multi_acc(output, labels, correct, total, multiclass_correct, multiclass_total, batch_size):
    _, predicted = torch.max(output.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    c = (predicted == labels).squeeze()
    for i in range(batch_size):
        label = labels[i]
        multiclass_correct[label] += c[i].item()
        multiclass_total[label] += 1

    return correct, total, multiclass_correct, multiclass_total


def train(config):
    data_loader = SolarDataLoader(config)
    train_loader = data_loader.train_loader
    val_loader = data_loader.val_loader
    # print(train_loader.__len__())

    # defining model
    model = CNN()
    lr = config['train']['optim']['lr']
    weight_decay = config['train']['optim']['weight_decay']
    momentum = config['train']['optim']['momentum']
    num_epochs = config['train']['epoch']
    step_size = config['scheduler']['step_size']
    log_dir = config['log']['log_dir']
    gamma = config['scheduler']['gamma']
    if config['train']['optim']['method'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr,  momentum=momentum,
                              weight_decay=weight_decay)
    # train_pipeline(optimizer, train_loader, model)
    scheduler = MultiStepLR(optimizer, milestones=step_size, gamma=gamma)

    # initialize best loss to a large value
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print('================= Started Epoch {}/{} ================='.format(epoch + 1, num_epochs))

        # train model on training dataset
        model.train()
        scheduler.step()
        loss_train = train_pipeline(optimizer, train_loader, model)
        print('Train Loss = {}'.format(loss_train))
        model.eval()
        loss_val, accuracy_total = val_pipeline(val_loader, model)
        print('Val Loss = {}'.format(loss_val))
        print('Accuracy of the network on the test images: %.3f' % (
                accuracy_total))
        if loss_val < best_loss:
            best_loss = loss_val
            torch.save(model.state_dict(), os.path.join(log_dir, 'model_epoch{}.pth'.format(epoch)))
            print("Copied best model weights!")
    print('End of training!')


if __name__ == "__main__":
    config = open_yaml(sys.argv[1])
    fire.Fire(train(config))