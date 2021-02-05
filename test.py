import numpy as np

# dataset_sizes = {'train': len(train_data_gen.dataset), 'valid': len(valid_data_gen.dataset)}
# print(dataset_sizes)

# for x, y in train_data_gen:
#     print("train:", x.shape)
#     print("train_label:", y.shape)
#     break

# print(train.class_to_idx)
# print(train.classes)

# for x, _ in train:
#     x.shape
# mean and std
# mean = [np.mean(x.numpy(), axis=(1, 2)) for x, _ in train]
# std = [np.std(x.numpy(), axis=(1, 2)) for x, _ in train]
#
# mean_val = np.mean([m[0] for m in mean])
# std_val = np.mean([s[0] for s in std])
#
# print(mean_val)
# print(std_val)
# 0.47684264
# 0.26336077



#Create data generators

# mean and std
# meanRGB = [np.mean(x.numpy(), axis=(1, 2)) for x, _ in train_ds]
# stdRGB = [np.std(x.numpy(), axis=(1, 2)) for x, _ in train_ds]
#
# meanR = np.mean([m[0] for m in meanRGB])
# meanG = np.mean([m[1] for m in meanRGB])
# meanB = np.mean([m[2] for m in meanRGB])
#
# stdR = np.mean([s[0] for s in stdRGB])
# stdG = np.mean([s[1] for s in stdRGB])
# stdB = np.mean([s[2] for s in stdRGB])
#
# print(meanR, meanG, meanB)
# print(stdR, stdG, stdB)

######### better way mean std
# mean = [np.mean(x.numpy()) for x, _ in train_loader]
# std = [np.std(x.numpy()) for x, _ in train_loader]
#
# mean_val = np.mean(mean)
# std_val = np.mean(std)
#
# print(mean_val)
# print(std_val)
# mean: 0.6512266
# std: 0.15099779