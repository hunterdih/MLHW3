from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random
import math
import time

random.seed(0)

M1 = [3, 3, 3]
C1 = [[2, 0, 0],
      [0, 2, 0],
      [0, 0, 2]]

M2 = [5, 5, 5]
C2 = [[2, 0, 0],
      [0, 2, 0],
      [0, 0, 2]]

M3 = [7, 7, 7]
C3 = [[1, 0, 0],
      [0, 1.9, 0],
      [0, 0, 1.7]]

M4 = [9, 9, 9]
C4 = [[3, 0, 0],
      [0, 1.5, 0],
      [0, 0, 0.5]]


def simple_mlp_model(x_train,
                     y_train,
                     x_test,
                     y_test,
                     layer_1_nodes=1,
                     activation_function='sigmoid',
                     optimizer_function='adam',
                     loss_function='sparse_categorical_crossentropy',
                     data_metrics=['accuracy'],
                     epoch_metric='val_loss',
                     time_limit=20,
                     optimal_train=False,
                     verbose=True):
    """

    :param x_train: Input training data, should be of shape ROWS=SAMPLE, COLUMNS=PARAMETERS
    :param y_train: Input training set labels, should be of shape ROWS=SAMPLE, COLUMNS=1(LABEL)
    :param x_test: Input test data, should be of shape ROWS=SAMPLE, COLUMNS=PARAMETERSr
    :param y_test: Input test set labels, should be of shape ROWS=SAMPLE, COLUMNS=1(LABEL)
    :param layer_1_nodes:
    :param activation_function: string containing the activation function
    :param optimizer_function:
    :param loss_function:
    :param data_metrics:
    :return y_results_train, y_results_test: Resulting labeled dataset
    :param epoch_metric: string containing metric to base epoch generations on
    :param verbose: Display output confusion matrix, loss and accuracy metrics
    :return:
    """

    # Need to determine the number of parameters in the dataset to optimize (this will determine the size of the nn)
    numParameters = x_train.shape[1]
    label_num = int(np.max(y_test)) + 1
    # Create the model
    model = Sequential()
    model.add(Dense(units=layer_1_nodes, kernel_initializer='random_uniform', activation='elu', input_dim=numParameters))
    # If there were multiple hidden layers to the network, they would be added here
    # Create output layer
    model.add(Dense(units=label_num, kernel_initializer='random_uniform', activation='softmax'))
    # Compile the model
    model.compile(optimizer=optimizer_function, loss=loss_function, metrics=data_metrics)

    # Initial Train
    history = model.fit(x_train, y_train, validation_split=0.15, batch_size=150, epochs=50, verbose=1)
    start_time = time.time()
    accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    validation_loss = history.history['val_loss']
    epochs = 0
    difference = abs(history.history[epoch_metric][0] - history.history[epoch_metric][9])
    epochs += 10
    print(f'{epochs=}')


    # Train until the accuracy difference between 10 epochs is less than 0.1% or n minutes has elapsed
    elapsed_time = 0
    if optimal_train:
        elapsed_time = time.time() - start_time
        while difference >= 0.0005 and elapsed_time < 60 * time_limit or epochs < 50:
            history = model.fit(x_train, y_train, validation_split=0.15, batch_size=150, epochs=10, verbose=1)
            accuracy.extend(history.history['accuracy'])
            validation_accuracy.extend(history.history['val_accuracy'])
            loss.extend(history.history['loss'])
            validation_loss.extend(history.history['val_loss'])
            difference = abs(history.history[epoch_metric][0] - history.history[epoch_metric][9])
            elapsed_time = time.time() - start_time
            epochs += 10
            print(f'{epochs=}')

    y_results_test = model.predict(x_test)
    print(history.history.keys())
    # summarize history for accuracy
    fig0 = plt.figure(0)
    plt.plot(accuracy)
    plt.plot(validation_accuracy)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    fig1 = plt.figure(1)
    # summarize history for loss
    plt.plot(loss)
    plt.plot(validation_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    fig2 = plt.figure(2)
    y_results_test = pd.DataFrame(y_results_test)
    y_results_test = y_results_test.idxmax(axis=1)
    cm = confusion_matrix(y_test, y_results_test, normalize='true')
    plt.imshow(cm, cmap='BuPu')
    for (i, j), label in np.ndenumerate(cm):
        plt.text(j, i, str(round(label, 4)), ha='center', va='center')
    plt.colorbar
    error = round(np.sum(1 - cm.diagonal()) / cm.shape[0], 4)
    plt.ylabel('True Label')
    plt.xlabel(f'Predicted Label, Error = {error}')
    if verbose:
        plt.show()
    fig0.clear()
    fig1.clear()
    fig2.clear()
    return max(accuracy), max(validation_accuracy), min(loss), min(validation_loss), epochs, error, y_results_test, model


def optimal_classifier(x_train, x_test, y_train, y_test):
    gnb = GaussianNB()
    y_pred = gnb.fit(x_train, y_train).predict(x_test)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    error = round(np.sum(1 - cm.diagonal()) / cm.shape[0], 4)
    return error, y_pred


def n_folds_calc(dataset, folds):
    '''

    :param dataset: Dataframe containing the entire dataset
    :param folds: Number of folds
    :return: Returns a list of indexes equal to the number of folds
    '''
    use_dataframe = dataset

    # Determine the number of entries in the dataset
    entries = use_dataframe.shape[0]
    samples_per_fold = math.floor(entries / folds)

    # Create index list
    entries_list = list(range(0, entries))

    # Shuffle the entries list to randomize sample selection in N-Fold
    random.shuffle(entries_list)
    return_entries = []
    for fold in range(folds):
        return_entries.append(entries_list[0:samples_per_fold])
        entries_list = entries_list[samples_per_fold:]

    return return_entries


def n_folds_split(data, indexes, fold):
    # Get test and train samples
    # Get test and train labels
    use_dataframe = data
    tr = use_dataframe.drop(indexes[fold])
    tst = use_dataframe.loc[use_dataframe.index[indexes[fold]]]

    return tr, tst


def generate_data(samples):
    data = []
    labels = []
    for i in range(samples):

        choice = random.random()
        if choice < 0.25:
            data.append(np.random.multivariate_normal(M1, C1, 1))
            labels.append(0)
        elif choice >= 0.25 and choice < 0.50:
            data.append(np.random.multivariate_normal(M2, C2, 1))
            labels.append(1)
        elif choice >= 0.50 and choice < 0.75:
            data.append(np.random.multivariate_normal(M3, C3, 1))
            labels.append(2)
        elif choice >= 0.75:
            data.append(np.random.multivariate_normal(M4, C4, 1))
            labels.append(3)
    data = np.stack(data, axis=1)
    data = pd.DataFrame(data[0])
    data['label'] = labels

    return data


if __name__ == '__main__':
    print(f'Starting...')
    target_parameter = "label"
    n_folds_crossvalidate = True
    n_folds = 10
    perceptron_limit = 20
    test_dataset = generate_data(100)
    y = test_dataset[target_parameter]
    x = test_dataset.drop(target_parameter, axis=1)
    x = pd.DataFrame(MinMaxScaler().fit_transform(x.loc[:].values), columns=x.columns)

    opt_err, opt_y_pred = optimal_classifier(x, x, y, y)
    print(f'Optimal Minimum Error={opt_err}')

    samples_list = [100, 200, 500, 1000, 2000, 5000]
    sample_size_error_list = []
    sample_size_best_pred_list = []
    model_list = []

    for samples in samples_list:
        perceptron_error_list = []
        perceptron_y_pred_list = []
        fold_selection = []
        dataset = generate_data(samples)

        # N-folds estimator
        if n_folds_crossvalidate:

            # Iterate through number of perceptrons in the first layer
            for num_perceptrons in range(1, perceptron_limit + 1):

                index_list = n_folds_calc(dataset, n_folds)
                acc_list = []
                acc_val_list = []
                loss_list = []
                loss_val_list = []
                epoch_list = []
                err_list = []
                y_pred_list = []

                # Iterate through each fold
                for i in range(n_folds):
                    train, test = n_folds_split(dataset, index_list, i)

                    y_train = train[target_parameter]
                    y_test = test[target_parameter]

                    x_train = train.drop(target_parameter, axis=1)
                    x_train = pd.DataFrame(MinMaxScaler().fit_transform(x_train.loc[:].values), columns=x_train.columns)
                    x_test = test.drop(target_parameter, axis=1)
                    x_test = pd.DataFrame(MinMaxScaler().fit_transform(x_test.loc[:].values), columns=x_test.columns)

                    max_acc, max_acc_val, min_loss, min_loss_val, epoch_chosen, err, y_pred, _ = simple_mlp_model(np.asarray(x_train), y_train, np.asarray(x_test), y_test, verbose=False,
                                                                                                               layer_1_nodes=num_perceptrons)
                    acc_list.append(max_acc)
                    acc_val_list.append(max_acc_val)
                    loss_list.append(min_loss)
                    loss_val_list.append(min_loss_val)
                    epoch_list.append(epoch_chosen)
                    err_list.append(err)
                    y_pred_list.append(y_pred)

                index_chosen = err_list.index(min(err_list))
                perceptron_error_list.append(min(err_list))

                perceptron_y_pred_list.append(y_pred_list[index_chosen])
                fold_selection.append(err_list.index(min(err_list)))

            use_fold = perceptron_error_list.index(min(perceptron_error_list))
            print(f'Fold Chosen: {use_fold}')
            train, test = n_folds_split(dataset, index_list, use_fold)

            y_train = train[target_parameter]
            y_test = test[target_parameter]

            x_train = train.drop(target_parameter, axis=1)
            x_train = pd.DataFrame(MinMaxScaler().fit_transform(x_train.loc[:].values), columns=x_train.columns)
            x_test = test.drop(target_parameter, axis=1)
            x_test = pd.DataFrame(MinMaxScaler().fit_transform(x_test.loc[:].values), columns=x_test.columns)

            _1, _2, _3, _4, _5, _6, _7, model = simple_mlp_model(np.asarray(x_train), y_train, np.asarray(x_test), y_test, verbose=True, layer_1_nodes=num_perceptrons)

            fig3 = plt.figure(3)
            plt.plot(acc_list)
            plt.plot(acc_val_list)
            plt.title(f'N-Folds Results (Accuracy) for {samples} Samples')
            plt.ylabel('Accuracy')
            plt.xlabel('Fold')
            plt.legend(['accuracy', 'validation accuracy'], loc='upper left')

            fig4 = plt.figure(4)
            plt.plot(loss_list)
            plt.plot(loss_val_list)
            plt.title(f'N-Folds Results (Loss) for {samples} Samples')
            plt.ylabel('Loss')
            plt.xlabel('Fold')
            plt.legend(['loss', 'validation loss'], loc='upper left')

            fig5 = plt.figure(5)
            plt.plot(perceptron_error_list)
            plt.axhline(y=opt_err, color='r', linestyle='-')
            plt.title(f'Perceptron Error vs. Optimal Error for {samples} Samples')
            plt.ylabel('Loss')
            plt.xlabel('Perceptrons')
            plt.legend(['Perceptron Loss', 'Optimal Loss'], loc='upper left')
            plt.show()
            fig3.clear()
            fig4.clear()
            fig5.clear()

            index_chosen = perceptron_error_list.index(min(perceptron_error_list))
            sample_size_error_list.append(perceptron_error_list[index_chosen])
            sample_size_best_pred_list.append(perceptron_y_pred_list[index_chosen])
            model_list.append(model)

    fig6 = plt.figure(6)
    plt.plot(sample_size_error_list)
    plt.axhline(y=opt_err, color='r', linestyle='-')
    plt.title(f'Sample Size Error vs. Optimal Error')
    x_ticks = range(len(samples_list))
    plt.ylabel('Loss')
    plt.xlabel('Sample Size')
    plt.xticks(x_ticks, samples_list)
    plt.legend(['Sample Loss', 'Optimal Loss'], loc='upper left')

    index_chosen = sample_size_error_list.index(min(sample_size_error_list))
    model = model_list[index_chosen]
    print(f'Running Final Evaluation on 100000 Test Set...')

    y_pred = model.fit(test_dataset)
    y_test = test_dataset['label']

    fig7 = plt.figure(7)
    y_results_test = pd.DataFrame(y_pred)
    y_results_test = y_results_test.idxmax(axis=1)
    cm = confusion_matrix(y_test, y_results_test, normalize='true')
    plt.imshow(cm, cmap='BuPu')
    for (i, j), label in np.ndenumerate(cm):
        plt.text(j, i, str(round(label, 4)), ha='center', va='center')
    plt.colorbar
    error = round(np.sum(1 - cm.diagonal()) / cm.shape[0], 4)
    plt.ylabel('True Label')
    plt.xlabel(f'Predicted Label, Error = {error}')
    plt.show()
    print(f'Done...')
