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
from InfraredSolarModules.utils.metrics import generate_cm
from InfraredSolarModules.utils.plot_confusion_matrix import plot_confusion_matrix


def train_pipeline(optimizer, train_loader, model, device):

    # history of loss values in each epoch
    loss_history = {
        "train": []
    }

    # histroy of metric values in each epoch
    metric_history = {
        "train": []
    }
    for itr, batch in enumerate(train_loader):
        # if itr == 5:
        #     break
        image, labels = batch
        image, labels = image.to(device), labels.to(device)
        optimizer.zero_grad()
        pred = model(image)
        loss = F.nll_loss(pred, labels)
        loss.backward()
        optimizer.step()
        if itr % 20 == 0:
            print('Train iteration {} had loss {:.6f}'.format(itr, loss))

        loss_history["train"].append(loss)
    loss_total = torch.mean(torch.Tensor(loss_history["train"]))
    return loss_total


def val_pipeline(val_loader, model, device):
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
        all_pred = torch.tensor([], device=device)
        for itr, batch in enumerate(val_loader):
            image, labels = batch
            image, labels = image.to(device), labels.to(device)
            pred = model(image)
            all_pred = torch.cat(
                (all_pred, pred)
                , dim=0
            )
            batch_size = image.shape[0]
            loss = F.nll_loss(pred, labels)
            loss_history["val"].append(loss)
            correct, total, multiclass_correct, multiclass_total = \
                multi_acc(pred, labels, correct, total, multiclass_correct, multiclass_total, batch_size)
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


def train(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print('Running model on: ', device)
    data_loader = SolarDataLoader(config)
    train_loader = data_loader.train_loader
    val_loader = data_loader.val_loader
    # print(train_loader.__len__())

    # defining model
    model = CNN().to(device)
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
        loss_train = train_pipeline(optimizer, train_loader, model, device)
        print('Train Loss = {}'.format(loss_train))

        # validation mode
        model.eval()
        loss_val, accuracy_total, cm = val_pipeline(val_loader, model, device)
        plot_confusion_matrix(cm, val_loader.dataset.classes, normalize=False,
                              file_name=os.path.join(log_dir, 'confusion_matrix_epoch{}.png'.format(epoch)))
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
