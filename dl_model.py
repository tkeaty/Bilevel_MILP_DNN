import numpy as np
import pandas as pd
import torch
from collections import OrderedDict
import numpy
import math
from sklearn import preprocessing
import matplotlib.pyplot as plt
import gapsplit_grb
from gurobipy.gurobipy import read
import gap_fba


def check_loss(l_i, l_c):
    if l_c > 0.2 and math.fabs((l_i-l_c))/l_i < 0.2:
        return True
    return False


class PassThroughLayer(torch.nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.mask = None

    def forward(self, mask, input):
        self.mask = mask
        return torch.mul(input, mask)

    def backward(self, grad_output):
        return torch.mul(grad_output, self.mask)


class LinModelPassThrough(torch.nn.Module):
    def __init__(self, input_size, int_input_size, red_factor=2, dropout=0.0):
        super().__init__()

        e_layers = OrderedDict()
        self.init_layer = None
        self.pass_layer = PassThroughLayer(int_input_size)
        self.thru_layer = None
        i = input_size
        ct = 0
        sizes = []

        while i//red_factor > 0:
            if i == input_size:
                l = torch.nn.Linear(i, int_input_size, dtype=torch.float64)
                torch.nn.init.xavier_uniform_(l.weight)
                self.init_layer = l
                l1 = torch.nn.Linear(int_input_size, i//2, dtype=torch.float64)
                torch.nn.init.xavier_uniform_(l1.weight)
                self.thru_layer = l1
                if dropout > 0.0:
                    e_layers['Drop' + str(ct)] = torch.nn.Dropout(p=dropout)

            elif ct % 2 == 0 and i//(red_factor*2) > 1:
                l = torch.nn.Linear(i, i // red_factor, dtype=torch.float64)
                torch.nn.init.xavier_uniform_(l.weight)
                e_layers['Lin' + str(ct)] = l
                # e_layers['Drop' + str(ct)] = torch.nn.Dropout(p=0.5)
                e_layers['ReLU' + str(ct)] = torch.nn.ReLU()
            elif i // red_factor == 1:
                l = torch.nn.Linear(i, i // red_factor, dtype=torch.float64)
                torch.nn.init.xavier_uniform_(l.weight)
                e_layers['Lin' + str(ct)] = l
            else:
                l = torch.nn.Linear(i, i // red_factor, dtype=torch.float64)
                torch.nn.init.xavier_uniform_(l.weight)
                e_layers['Lin' + str(ct)] = l
                e_layers['ReLU' + str(ct)] = torch.nn.ReLU()

            sizes.append((i, i//red_factor))

            i = i//red_factor
            ct += 1

        self.encoder = torch.nn.Sequential(
            e_layers
        )

    def forward(self, x, ints):
        lin_pass = self.init_layer(x)
        int_pass = self.pass_layer(ints, lin_pass)
        thru_pass = self.thru_layer(int_pass)

        return self.encoder(thru_pass)


class LinModel(torch.nn.Module):
    def __init__(self, input_size, red_factor=2, dropout=0.0):
        super().__init__()

        e_layers = OrderedDict()
        i = input_size
        ct = 0
        sizes = []

        while i//red_factor > 0:
            if i == input_size:
                l = torch.nn.Linear(i, i // red_factor, dtype=torch.float64)
                torch.nn.init.xavier_uniform_(l.weight)
                e_layers['Lin' + str(ct)] = l
                if dropout > 0.0:
                    e_layers['Drop' + str(ct)] = torch.nn.Dropout(p=dropout)

            elif ct % 2 == 0 and i//(red_factor*2) > 1:
                l = torch.nn.Linear(i, i // red_factor, dtype=torch.float64)
                torch.nn.init.xavier_uniform_(l.weight)
                e_layers['Lin' + str(ct)] = l
                # e_layers['Drop' + str(ct)] = torch.nn.Dropout(p=0.5)
                e_layers['ReLU' + str(ct)] = torch.nn.ReLU()
            elif i // red_factor == 1:
                l = torch.nn.Linear(i, i // red_factor, dtype=torch.float64)
                torch.nn.init.xavier_uniform_(l.weight)
                e_layers['Lin' + str(ct)] = l
            else:
                l = torch.nn.Linear(i, i // red_factor, dtype=torch.float64)
                torch.nn.init.xavier_uniform_(l.weight)
                e_layers['Lin' + str(ct)] = l
                e_layers['ReLU' + str(ct)] = torch.nn.ReLU()

            sizes.append((i, i//red_factor))

            i = i//red_factor
            ct += 1

        self.encoder = torch.nn.Sequential(
            e_layers
        )

    def forward(self, x):
        return self.encoder(x)


def tuplize_samples(data):
    new_data = []
    for i in range(data.shape[0]):
        new_data.append([data[i, :-1], data[i, -1]])

    return new_data


def training_loop(samples, n_add, n_secondary, n_train, epochs, batch_size, inputs, objs, lr=0.001, drop=0.0, red_factor=2):

    train_samples = samples.iloc[:n_train]
    test_samples = samples.iloc[n_train:]

    data = np.vstack((train_samples.values, test_samples.values))

    scaler = preprocessing.StandardScaler()
    scaler.fit(data)

    scaled_train = scaler.transform(train_samples.values)
    scaled_test = scaler.transform(test_samples.values)

    train = []
    for i in range(scaled_train.shape[0]):
        train.append([scaled_train[i, :-1], scaled_train[i, -1]])

    test = []
    for i in range(scaled_test.shape[0]):
        test.append([scaled_test[i, :-1], scaled_test[i, -1]])

    # Model Initialization
    model = LinModel(input_size=scaled_train.shape[1] - 1, dropout=drop, red_factor=red_factor)

    # Validation using MSE Loss function
    loss_function = torch.nn.MSELoss()

    # Using an Adam Optimizer with lr = 0.1
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=1e-8)

    losses = []
    test_losses = []

    train_loader = torch.utils.data.DataLoader(dataset=train,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test,
                                              batch_size=batch_size,
                                              shuffle=True)

    for epoch in range(epochs):
        # print('Epoch %s' % epoch, end='\r')
        # if epoch == 6:
        #     l_0 = e_loss[-1]
        # elif epoch > 25:
        #     if check_loss(l_0, e_loss[-1]):
        #         new_samples = gapsplit_grb.sample(fpath, n_add, n_secondary=n_secondary)
        #         new_samples_s = scaler.transform(new_samples.values)
        #
        #         new_samples_load = tuplize_samples(new_samples_s)
        #         train += new_samples_load
        #
        #         print('Train set: %i' % len(train))
        #
        #         train_loader = torch.utils.data.DataLoader(dataset=train,
        #                                                    batch_size=batch_size,
        #                                                    shuffle=True)

        e_loss = []
        model.train()

        for pts, labels in train_loader:  # (pts, _)
            # Output of Autoencoder
            pred = model(pts)

            # Calculating the loss function
            loss = loss_function(pred, torch.reshape(labels, (len(labels), 1)))

            # The gradients are set to zero,
            # the gradient is computed and stored.
            # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Storing the losses in a list for plotting
            e_loss.append(loss.item())
            # outputs.append((epoch, pts, reconstructed))

        test_loss = []
        model.eval()

        for pts, labels in test_loader:
            pred = model(pts)

            np_pred = pred.detach().numpy()
            np_pts = pts.numpy()

            loss = loss_function(pred, torch.reshape(labels, (len(labels), 1)))
            test_loss.append(loss.item())

        optimizer.zero_grad()
        test_losses.append(np.mean(test_loss))
        losses.append(np.mean(e_loss))

    # Bi-level step
    scaled_inputs = scaler.transform(np.hstack((inputs, objs.reshape(-1,1))))

    obj_test = []
    for i in range(inputs.shape[0]):
        obj_test.append([scaled_inputs[i, :-1], scaled_inputs[i, -1]])

    obj_loader = torch.utils.data.DataLoader(dataset=obj_test,
                                               batch_size=1,
                                               shuffle=False)

    obj_losses = []
    pred_objs = []
    true_objs = []
    model.eval()
    for pt, obj in obj_loader:
        pred = model(pt)
        pred_objs.append(pred.detach().numpy()[0])
        true_objs.append(obj.numpy())

        loss = loss_function(pred, torch.reshape(obj, (1, 1)))
        obj_losses.append(loss.item())

    pred_objs = np.asarray(pred_objs)
    true_objs = np.asarray(true_objs)

    right_obj = 0

    if np.argmin(true_objs) == np.argmin(pred_objs):
        right_obj = 1

    return right_obj, np.argmin(pred_objs), np.asarray(losses), np.asarray(test_losses)


def training_loop_int(samples, int_samples, n_train, epochs, inputs, int_inputs, objs, batch_size=64, lr=0.001, drop=0.0, red_factor=2):

    train_samples = samples.iloc[:n_train]
    test_samples = samples.iloc[n_train:]

    train_int_samples = int_samples.iloc[:n_train].values
    test_int_samples = int_samples.iloc[n_train:].values

    data = np.vstack((train_samples.values, test_samples.values))

    scaler = preprocessing.StandardScaler()
    scaler.fit(data)

    scaled_train = scaler.transform(train_samples.values)
    scaled_test = scaler.transform(test_samples.values)

    train = []
    for i in range(scaled_train.shape[0]):
        train.append([scaled_train[i, :-1], train_int_samples[i, :], scaled_train[i, -1]])

    test = []
    for i in range(scaled_test.shape[0]):
        test.append([scaled_test[i, :-1], test_int_samples[i, :], scaled_test[i, -1]])

    # Model Initialization
    model = LinModelPassThrough(input_size=scaled_train.shape[1] - 1, int_input_size=int_samples.shape[1], dropout=drop, red_factor=red_factor)

    # Validation using MSE Loss function
    loss_function = torch.nn.MSELoss()

    # Using an Adam Optimizer with lr = 0.1
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=1e-8)

    losses = []
    test_losses = []

    train_loader = torch.utils.data.DataLoader(dataset=train,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test,
                                              batch_size=batch_size,
                                              shuffle=True)

    for epoch in range(epochs):
        # print('Epoch %s' % epoch, end='\r')
        # if epoch == 6:
        #     l_0 = e_loss[-1]
        # elif epoch > 25:
        #     if check_loss(l_0, e_loss[-1]):
        #         new_samples = gapsplit_grb.sample(fpath, n_add, n_secondary=n_secondary)
        #         new_samples_s = scaler.transform(new_samples.values)
        #
        #         new_samples_load = tuplize_samples(new_samples_s)
        #         train += new_samples_load
        #
        #         print('Train set: %i' % len(train))
        #
        #         train_loader = torch.utils.data.DataLoader(dataset=train,
        #                                                    batch_size=batch_size,
        #                                                    shuffle=True)

        e_loss = []
        model.train()

        for pts, ints, labels in train_loader:  # (pts, _)
            # Output of Autoencoder
            pred = model(pts, ints)

            # Calculating the loss function
            loss = loss_function(pred, torch.reshape(labels, (len(labels), 1)))

            # The gradients are set to zero,
            # the gradient is computed and stored.
            # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Storing the losses in a list for plotting
            e_loss.append(loss.item())
            # outputs.append((epoch, pts, reconstructed))

        test_loss = []
        model.eval()

        for pts, ints, labels in test_loader:
            pred = model(pts, ints)

            np_pred = pred.detach().numpy()
            np_pts = pts.numpy()

            loss = loss_function(pred, torch.reshape(labels, (len(labels), 1)))
            test_loss.append(loss.item())

        optimizer.zero_grad()
        test_losses.append(np.mean(test_loss))
        losses.append(np.mean(e_loss))

    # Bi-level step
    scaled_inputs = scaler.transform(np.hstack((inputs, objs.reshape(-1,1))))

    obj_test = []
    for i in range(inputs.shape[0]):
        obj_test.append([scaled_inputs[i, :-1], int_inputs[i, :], scaled_inputs[i, -1]])

    obj_loader = torch.utils.data.DataLoader(dataset=obj_test,
                                               batch_size=1,
                                               shuffle=False)

    obj_losses = []
    pred_objs = []
    true_objs = []
    model.eval()
    for pt, int_input, obj in obj_loader:
        pred = model(pt, int_input)
        pred_objs.append(pred.detach().numpy()[0])
        true_objs.append(obj.numpy())

        loss = loss_function(pred, torch.reshape(obj, (1, 1)))
        obj_losses.append(loss.item())

    pred_objs = np.asarray(pred_objs)
    true_objs = np.asarray(true_objs)

    right_obj = 0

    if np.argmin(true_objs) == np.argmin(pred_objs):
        right_obj = 1

    return right_obj, np.argmin(pred_objs), np.asarray(losses), np.asarray(test_losses)


def run_experiment(samples, inputs, objs, n_samples, n_train, n_secondary=3, dropout=0.0, batch_size=32, learning_rate=0.001, red_factor=2, label=''):
    # samples = gapsplit_grb.sample(fpath, n_samples, n_secondary=n_secondary)
    # inputs, objs = gap_fba.sample(fpath)
    epochs = 100
    runs = 100
    test_losses = np.zeros((runs, epochs))
    train_losses = np.zeros((runs, epochs))
    guessed_objs = np.zeros(runs + 1)

    n_right = 0
    for r in range(runs):
        right_obj, guessed_obj, train, test = training_loop(samples, 1000, 3, n_train, epochs, batch_size, inputs, objs, drop=dropout, lr=learning_rate, red_factor=red_factor)
        train_losses[r, :] = train
        test_losses[r, :] = test
        guessed_objs[r] = guessed_obj

        n_right += right_obj

    guessed_objs[-1] = np.argmin(objs)
    print(n_right)
    print(float(n_right)/runs)

    np.savetxt(label + 'bilevel_pred_objs.csv', guessed_objs, delimiter=',')
    np.savetxt(label + 'train_losses.csv', train_losses, delimiter=',')
    np.savetxt(label + 'test_losses.csv', test_losses, delimiter=',')


def run_experiment_int(samples, int_samples, inputs, int_inputs, objs, label, dropout=0.0, batch_size=64, learning_rate=0.0001, red_factor=2):
    # samples = gapsplit_grb.sample(fpath, n_samples, n_secondary=n_secondary)
    # inputs, objs = gap_fba.sample(fpath)
    epochs = 50
    runs = 10
    test_losses = np.zeros((runs, epochs))
    train_losses = np.zeros((runs, epochs))
    guessed_objs = np.zeros(runs + 1)

    n_right = 0
    for r in range(runs):
        right_obj, guessed_obj, train, test = training_loop_int(samples, int_samples, 5000, epochs, inputs, int_inputs, objs, drop=dropout, batch_size=batch_size, lr=learning_rate, red_factor=red_factor)
        train_losses[r, :] = train
        test_losses[r, :] = test
        guessed_objs[r] = guessed_obj

        n_right += right_obj

    guessed_objs[-1] = np.argmin(objs)
    print(n_right)
    print(float(n_right)/runs)

    np.savetxt(label + 'bilevel_pred_objs.csv', guessed_objs, delimiter=',')
    np.savetxt(label + 'train_losses.csv', train_losses, delimiter=',')
    np.savetxt(label + 'test_losses.csv', test_losses, delimiter=',')


def run_experiment_int_all(samples, int_samples, inputs, int_inputs, objs, label, dropout=0.0, batch_size=64, learning_rate=0.0001, red_factor=2):
    # samples = gapsplit_grb.sample(fpath, n_samples, n_secondary=n_secondary)
    # inputs, objs = gap_fba.sample(fpath)
    epochs = 100
    runs = 100
    test_losses = np.zeros((runs, epochs))
    train_losses = np.zeros((runs, epochs))
    guessed_objs = np.zeros(runs + 1)

    n_right = 0
    for r in range(runs):
        right_obj, guessed_obj, train, test = training_loop_int(samples, int_samples, 5000, 100, inputs, int_inputs, objs, drop=dropout, batch_size=batch_size, lr=learning_rate, red_factor=red_factor)
        train_losses[r, :] = train
        test_losses[r, :] = test
        guessed_objs[r] = guessed_obj

        n_right += right_obj

    guessed_objs[-1] = np.argmin(objs)
    print(n_right)
    print(float(n_right)/runs)

    np.savetxt(label + 'bilevel_pred_objs.csv', guessed_objs, delimiter=',')
    np.savetxt(label + 'train_losses.csv', train_losses, delimiter=',')
    np.savetxt(label + 'test_losses.csv', test_losses, delimiter=',')


if __name__ == '__main__':
    # (samples, n_add, n_secondary, n_train, epochs, batch_size, inputs, objs, lr=0.001, drop=0.0, red_factor=2)
    samples, _ = gapsplit_grb.sample('ind750.lp', 1300, n_secondary=3)
    samples.to_csv('ind750_test_obj_constr.csv', sep=',')
    inputs, _, objs = gap_fba.sample('ind750.lp')
    epochs = 50
    runs = 10
    test_losses = np.zeros((runs, epochs))
    train_losses = np.zeros((runs, epochs))
    guessed_objs = np.zeros(runs+1)

    n_right = 0
    for r in range(runs):
        print(r)
        right_obj, guessed_obj, train, test = training_loop(samples, 1000, 3, 1000, epochs, 64, inputs, objs, red_factor=3)
        train_losses[r, :] = train
        test_losses[r, :] = test
        guessed_objs[r] = guessed_obj

        n_right += right_obj

    guessed_objs[-1] = np.argmin(objs)
    print(n_right)
    print(float(n_right)/runs)

    # Plotting
    # train_mean = np.mean(train_losses, axis=0)
    # train_std = np.std(train_losses, axis=0)
    #
    # test_mean = np.mean(test_losses, axis=0)
    # test_std = np.std(test_losses, axis=0)
    #
    # plt.figure(figsize=(10, 10))
    # plt.style.use('fivethirtyeight')
    # plt.xlabel('Iterations')
    # plt.ylabel('Loss')
    # plt.show()
    # path = 'iSMU_srfba.mps'
    # samples, int_samples = gapsplit_grb.sample(path, 5300, n_secondary=10)
    # inputs, int_inputs, objs = gap_fba.sample(path)
    #
    # dropouts = [0.0, 0.2]
    # batch_sizes = [32, 64]
    # red_factors = [2, 3]
    # lrs = [0.0001, 0.001]
    #
    # print('drop')
    # for dropout in dropouts:
    #     label = 'results/data_2/' + path[:-4] + '_std_' + '_drop_' + str(dropout)
    #     run_experiment_int(samples, int_samples, inputs, int_inputs, objs, label, dropout=dropout, batch_size=64, learning_rate=0.001)
    #
    # print('batch')
    # for batch in batch_sizes:
    #     label = 'results/data_2/' + path[:-4] + '_std_' + '_batch_' + str(batch)
    #     run_experiment_int(samples, int_samples, inputs, int_inputs, objs, label, dropout=0.0, batch_size=batch,
    #                    learning_rate=0.001)
    #
    # print('rf')
    # for rf in red_factors:
    #     label = 'results/data_2/' + path[:-4] + '_std_' + '_rf_' + str(rf)
    #     run_experiment_int(samples, int_samples, inputs, int_inputs, objs, label, dropout=0.0, batch_size=64,
    #                    learning_rate=0.001, red_factor=rf)
    #
    # print('lr')
    # for lr in lrs:
    #     label = 'results/data_2/' + path[:-4] + '_std_' + '_lr_' + str(lr)
    #     run_experiment_int(samples, int_samples, inputs, int_inputs, objs, label, dropout=0.0, batch_size=64, learning_rate=lr, red_factor=2)

    a = None

    # paths = [
    #     # 'Ecoli_core_model.lp',
    #     'ind750.lp',
    #     'Ecoli_core_model.lp',
    #     'pao.lp'
    # ]
    #
    # dropouts = [0.0, 0.2, 0.4, 0.6, 0.8]
    # batch_sizes = [4, 8, 16, 32, 64]
    # red_factors = [2, 3, 4]
    # lrs = [0.0001, 0.001, 0.01, 0.1]
    # sample_sizes = [1000, 5000, 10000, 15000]
    # cts_sampling = [100, 500, 1000]
    #
    # for path in paths:
    #     samples, _ = gapsplit_grb.sample(path, 5300, n_secondary=3)
    #     inputs, _, objs = gap_fba.sample(path)

        # print('drop')
        # for dropout in dropouts:
        #     label = 'results/data_2/' + path[:-3] + '_drop_' + str(dropout).replace('.', '-')
        #     run_experiment(samples, inputs, objs, 5300, 5000, n_secondary=3, dropout=dropout, batch_size=32, learning_rate=0.001, label=label)
        #
        # print('batch')
        # for batch in batch_sizes:
        #     label = 'results/data_2/' + path[:-3] + '_batch_' + str(batch)
        #     run_experiment(samples, inputs, objs, 5300, 5000, n_secondary=3, dropout=0.0, batch_size=batch, learning_rate=0.001,
        #                    label=label)

        # print('rf')
        # for rf in red_factors:
        #     label = 'results/data_2/' + path[:-3] + '_rf_' + str(rf)
        #     run_experiment(samples, inputs, objs, 5300, 5000, n_secondary=3, dropout=0.0, batch_size=32, learning_rate=0.001, red_factor=rf,
        #                    label=label)

        # print('lr')
        # for lr in lrs:
        #     label = 'results/data_2/' + path[:-3] + '_lr_' + str(lr)
        #     run_experiment(samples, inputs, objs, 5300, 5000, n_secondary=3, dropout=0.0, batch_size=32, learning_rate=lr,
        #                    red_factor=2,
        #                    label=label)

        # print('ss')
        # for ss in sample_sizes:
        #     label = 'results/data/' + path[:-3] + '_samples_' + str(ss)
        #     run_experiment(path, ss, ss+300, n_secondary=3, dropout=0.0, batch_size=16, learning_rate=0.001,
        #                    red_factor=2,
        #                    label=label)
    # np.savetxt('bilevel_pred_objs.csv', guessed_objs, delimiter=',')
    # np.savetxt('cts_sample_debug_ind750.csv', train_losses, delimiter=',')
    #
    # # Plotting the last 100 values
    # plt.plot(range(len(train_mean)), train_mean, c='r')
    # plt.fill_between(range(len(train_mean)), train_mean - train_std, train_mean + train_std, color='r', alpha=0.15)
    #
    # plt.plot(range(len(test_mean)), test_mean, c='g')
    # plt.fill_between(range(len(test_mean)), test_mean - test_std, test_mean + test_std, color='g', alpha=0.15)
    #
    # plt.legend(['Training mean', '+/- std', 'Test', '+/- std'])
    # plt.title('Pseudomonas model, 5k samples, 100 runs')
    # plt.savefig('pao_5000train_300test_100runs.jpg')

    # bi-level LP value testing
    # inputs, objs = gap_fba.sample('Ecoli_core_model.lp')
    #
    # scaled_inputs = scaler.transform(np.hstack((inputs, objs.reshape(-1,1))))
    #
    # obj_test = []
    # for i in range(inputs.shape[0]):
    #     obj_test.append([scaled_inputs[i, :-1], scaled_inputs[i, -1]])
    #
    # obj_loader = torch.utils.data.DataLoader(dataset=obj_test,
    #                                            batch_size=1,
    #                                            shuffle=False)
    #
    # obj_losses = []
    # pred_objs = []
    # true_objs = []
    # for pt, obj in obj_loader:
    #     pred = model(pt)
    #     pred_objs.append(pred.detach().numpy()[0])
    #     true_objs.append(obj.numpy())
    #
    #     loss = loss_function(pred, torch.reshape(obj, (1, 1)))
    #     obj_losses.append(loss.item())
    #
    # pred_objs = np.asarray(pred_objs)
    # true_objs = np.asarray(true_objs)
    #
    # print(np.argmin(true_objs))
    # print(np.argmin(pred_objs))



    # Defining the Plot Style
