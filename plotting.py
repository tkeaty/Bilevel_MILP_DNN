import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# assign directory
directory = 'results/data_2'

# data = []
# labels = []
#
# model = 'pao'
# test = 'rf'
# sample_type = 'bilevel'
#
# # iterate over files in
# # that directory
# for filename in os.listdir(directory):
#     f = os.path.join(directory, filename)
#     # checking if it is a file
#     if os.path.isfile(f) and model in f and test in f and sample_type in f and 'std' not in f:
#         temp_data = np.loadtxt(f, delimiter=',')
#         data.append(temp_data)
#         f_segs = f.split('_')
#
#         for seg in f_segs:
#             if sample_type in seg:
#                 labels.append(int(seg.strip(sample_type).replace('-', '.')))
#
# n_right = []
#
# sorted_labels = np.argsort(labels)
#
# labels = [labels[i] for i in sorted_labels]
# data = [data[i] for i in sorted_labels]
#
# for d in data:
#     true = d[-1]
#     pred = d[:-1]
#     n_right.append(np.sum(np.where(pred == true, 1, 0))/100)
#
# plt.figure()
# plt.plot(labels, n_right)
# # plt.xscale('log')
# plt.xlabel('Layer size reduction factor')
# plt.xticks(labels, labels)
# plt.ylabel('Accuracy')
# plt.title('Accuracy of bi-level objective predictions, PAO model')
# plt.savefig('results/figs_2/'+model+'_'+test+'_'+sample_type+'.jpeg')




# iterate over files in
# that directory
# .replace('-','.')
a = None

# train_data = []
# test_data = []
# test_labels = []
# train_labels = []
#
# model = 'ind750'
# test = 'rf'
# sample_type = 'losses'
# for filename in os.listdir(directory):
#     f = os.path.join(directory, filename)
#     # checking if it is a file
#     if os.path.isfile(f) and model in f and test in f and sample_type in f:
#         temp_data = np.loadtxt(f, delimiter=',')
#
#         if 'test' in f:
#             test_data.append(temp_data)
#             f_segs = f.split('_')
#
#             for seg in f_segs:
#                 if 'test' in seg:
#                     test_labels.append(int(seg.strip('test').replace('-','.')))
#
#         if 'train' in f:
#             train_data.append(temp_data)
#             f_segs = f.split('_')
#
#             for seg in f_segs:
#                 if 'train' in seg:
#                     train_labels.append(int(seg.strip('train').replace('-','.')))
#
#
# n_right = []
#
# sorted_test_labels = np.argsort(test_labels)
# sorted_train_labels = np.argsort(train_labels)
#
# train_labels = [train_labels[i] for i in sorted_train_labels]
# train_data = [train_data[i] for i in sorted_train_labels]
#
# test_labels = [test_labels[i] for i in sorted_test_labels]
# test_data = [test_data[i] for i in sorted_test_labels]
#
# print(train_labels)
# print(test_labels)
#
#
# train_final = []
# for train in train_data:
#     train_final.append(train[:, -1].flatten())
#
# test_final = []
# for test in test_data:
#     test_final.append(test[:, -1].flatten())
#
# plt.figure()
# plt.boxplot(train_final)
# train_labels = [0] + train_labels
# plt.xticks(range(len(train_labels)), train_labels)
# plt.xlabel('Layer size reduction factor')
# plt.ylabel('Final training loss')
# plt.title('Final training loss over 100 epochs, Yeast model')
# plt.savefig('results/figs_2/'+model+'_'+'rf_'+'train_loss.jpeg')
# plt.close()
#
# plt.figure()
# plt.boxplot(test_final)
# test_labels = [0] + test_labels
# plt.xticks(range(len(test_labels)), test_labels)
# plt.xlabel('Layer size reduction factor')
# plt.ylabel('Final test loss')
# plt.title('Final test loss over 100 epochs, Yeast model')
# plt.savefig('results/figs_2/'+model+'_'+'rf_'+'test_loss.jpeg')
# plt.close()
