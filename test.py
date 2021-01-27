# dataset_sizes = {'train': len(train_data_gen.dataset), 'valid': len(valid_data_gen.dataset)}
# print(dataset_sizes)
# #
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

