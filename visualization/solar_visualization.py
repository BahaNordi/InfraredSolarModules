import fire
import sys
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from InfraredSolarModules.config.yaml_reader import open_yaml
from InfraredSolarModules.data.dataloader import SolarTestDataLoader
from InfraredSolarModules.models.model import CNN
from InfraredSolarModules.models.resnet_customized import resnet20, resnet32
from InfraredSolarModules.models.densenet_customized import densenet121
from InfraredSolarModules.utils.metrics import generate_cm
import matplotlib.pyplot as plt

FASHION_LABELS = {
    0: 'Cell',
    1: 'Cell-M',
    2: 'Crack',
    3: 'Diode',
    4: 'Diode-M',
    5: 'Hotspot',
    6: 'Hotspot-M',
    7: 'Normal',
    8: 'Offline',
    9: 'Shadow',
    10: 'Soil',
    11: 'Veg'
}


def inference(config):
    data_loader = SolarTestDataLoader(config)
    test_loader_vis = data_loader.test_loader_visualisation
    test_loader = data_loader.test_loader
    class_to_consider = 4
    idx = torch.tensor(test_loader.dataset.targets) == class_to_consider
    random_samples = 20
    random_indices = np.random.choice(np.flatnonzero(idx), random_samples)

    dataset_test_vis = torch.utils.data.dataset.Subset(test_loader_vis.dataset, random_indices)
    subset_test_vis = torch.utils.data.DataLoader(dataset_test_vis, batch_size=random_samples, shuffle=False)
    dataset_test = torch.utils.data.dataset.Subset(test_loader.dataset, random_indices)
    subset_test = torch.utils.data.DataLoader(dataset_test, batch_size=random_samples, shuffle=False)

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
        image, labels = next(iter(subset_test))
        image_vis, labels_vis = next(iter(subset_test_vis))
        image, labels = image.to(device), labels.to(device)
        pred = model(image)
        pred = F.softmax(pred, -1)
        max_pred, predicted = torch.max(pred.data, 1)
        orig_class_prob = pred[:, class_to_consider]
        display_sample(image_vis.cpu().numpy(), labels_vis.cpu().numpy(), max_pred, predicted, orig_class_prob,
                       plot_title='Sample images of the "%s" class' % FASHION_LABELS[class_to_consider],
                       num_rows=random_samples//5, num_cols=5)

        print("")


def display_sample(sample_images, sample_labels, sample_prob=None, sample_predictions=None, orig_class_prob=None,
                   num_rows=5, num_cols=10, plot_title=None, fig_size=None):
    """ display a random selection of images & corresponding labels, optionally with predictions
        The display is laid out in a grid of num_rows x num_col cells
        If sample_predictions are provided, then each cell's title displays the prediction
        (if it matches actual) or actual/prediction if there is a mismatch
    """
    import seaborn as sns
    assert sample_images.shape[0] == num_rows * num_cols

    with sns.axes_style("whitegrid"):
        sns.set_context("notebook", font_scale=1.1)
        sns.set_style({"font.sans-serif": ["Verdana", "Arial", "Calibri", "DejaVu Sans"]})

        f, ax = plt.subplots(num_rows, num_cols, figsize=((15, 12) if fig_size is None else fig_size),
            gridspec_kw={"wspace": 0.7, "hspace": 0.30}, squeeze=True)

        for r in range(num_rows):
            for c in range(num_cols):
                image_index = r * num_cols + c
                ax[r, c].axis("off")
                # show selected image
                ax[r, c].imshow(np.transpose(sample_images[image_index], (1, 2, 0)), cmap="Greys")
                # ax[r, c].imshow(sample_images[image_index], cmap="Greys")

                if sample_predictions is None:
                    # show the actual labels in the cell title
                    title = ax[r, c].set_title("%s" % FASHION_LABELS[sample_labels[image_index].item()])
                    plt.setp(title)
                else:
                    # else check if prediction matches actual value
                    true_label = sample_labels[image_index]
                    pred_label = sample_predictions[image_index]
                    pred_prob = sample_prob[image_index]
                    orig_prob = orig_class_prob[image_index]
                    prediction_matches_true = (sample_labels[image_index] == sample_predictions[image_index])
                    if prediction_matches_true:
                        # if actual == prediction, cell title is prediction shown in green font
                        title = '%s (p=%.2f)' % (FASHION_LABELS[true_label.item()], pred_prob.item())
                        title_color = 'g'
                    else:
                        # if actual != prediction, cell title is actua/prediction in red font
                        title = '%s (p=%.2f) \n %s (p=%.2f)' % (FASHION_LABELS[true_label.item()],
                                                      orig_prob.item(),
                                                      FASHION_LABELS[pred_label.item()],
                                                      pred_prob.item())
                        title_color = 'r'
                    # display cell title
                    title = ax[r, c].set_title(title)
                    plt.setp(title, color=title_color)
        # set plot title, if one specified
        if plot_title is not None:
            f.suptitle(plot_title)

        plt.show()
        plt.close()


if __name__ == "__main__":
    config = open_yaml(sys.argv[1])
    fire.Fire(inference(config))

