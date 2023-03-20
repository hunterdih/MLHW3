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
import os

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

OUTPATH = r'D:\MLResults\HW3\Problem1'


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
                     time_limit=5,
                     optimal_train=False,
                     fold_number=0,
                     sample_number=0,
                     save_figs=False,
                     warm_epoch_count=50,
                     batch_size=100,
                     verbose=True):
    """

    :param fold_number:
    :param time_limit:
    :param optimal_train:
    :param sample_number:
    :param save_figs:
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
    # label_num = int(np.max(y_test)) + 1 # Dynamic way to determine number of labels
    label_num = 4
    # Create the model
    model = Sequential()
    model.add(Dense(units=layer_1_nodes, kernel_initializer='random_uniform', activation='elu', input_dim=numParameters))
    # If there were multiple hidden layers to the network, they would be added here
    # Create output layer
    model.add(Dense(units=label_num, kernel_initializer='random_uniform', activation='softmax'))
    # Compile the model
    model.compile(optimizer=optimizer_function, loss=loss_function, metrics=data_metrics)

    # Initial Train
    history = model.fit(x_train, y_train, validation_split=0.15, batch_size=batch_size, epochs=warm_epoch_count, verbose=0)
    start_time = time.time()
    accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    validation_loss = history.history['val_loss']
    epochs = 0
    difference = abs(history.history[epoch_metric][0] - history.history[epoch_metric][9])
    epochs += warm_epoch_count
    print(f'{epochs=}')

    # Train until the accuracy difference between 10 epochs is less than 0.1% or n minutes has elapsed
    elapsed_time = 0
    if optimal_train:
        elapsed_time = time.time() - start_time
        while difference >= 0.0005 and elapsed_time < 60 * time_limit or epochs < 50:
            history = model.fit(x_train, y_train, validation_split=0.15, batch_size=batch_size, epochs=10, verbose=0)
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
    plt.title(f'Model Accuracy {layer_1_nodes} Perceptrons')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    fig1 = plt.figure(1)
    # summarize history for loss
    plt.plot(loss)
    plt.plot(validation_loss)
    plt.title(f'Model Loss {layer_1_nodes} Perceptrons')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    fig2 = plt.figure(2)
    y_results_test = pd.DataFrame(y_results_test)
    y_results_test = y_results_test.idxmax(axis=1)
    cm = confusion_matrix(y_test, y_results_test, normalize='true')
    plt.imshow(cm, cmap='BuPu')
    for (i, j), label in np.ndenumerate(cm):
        plt.text(j, i, str(round(label, 4)), ha='center', va='center')
    plt.colorbar
    error = round(np.sum(1 - cm.diagonal()) / cm.shape[0], 4)
    plt.title(f'Confusion Matrix {layer_1_nodes} Perceptrons')
    plt.ylabel('True Label')
    plt.xlabel(f'Predicted Label, Error = {error}')
    if verbose:
        plt.show()
    if save_figs:
        fig0.savefig(OUTPATH + f'/{num_perceptrons}_perceptrons/{sample_number}samples__{fold_number}fold_acc')
        fig1.savefig(OUTPATH + f'/{num_perceptrons}_perceptrons/{sample_number}samples__{fold_number}fold_loss')
        fig2.savefig(OUTPATH + f'/{num_perceptrons}_perceptrons/{sample_number}samples__{fold_number}fold_cm')

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
    optimal_train = False
    n_folds_crossvalidate = True
    verb = False
    n_folds = 2
    perceptron_limit = 2
    epochs = 200
    test_dataset = generate_data(100000)
    y = test_dataset[target_parameter]
    x = test_dataset.drop(target_parameter, axis=1)
    x = pd.DataFrame(MinMaxScaler().fit_transform(x.loc[:].values), columns=x.columns)

    opt_err, opt_y_pred = optimal_classifier(x, x, y, y)
    print(f'Optimal Minimum Error={opt_err}')

    samples_list = [100, 200, 500, 1000, 2000, 5000]
    sample_size_error_list = []
    sample_size_best_pred_list = []
    sample_size_perceptrons = []

    if not os.path.exists(OUTPATH + '/results'):
        os.makedirs(OUTPATH + '/results')

    y_test_100k = test_dataset[target_parameter]
    x_test_100k = test_dataset.drop(target_parameter, axis=1)
    x_test_100k = pd.DataFrame(MinMaxScaler().fit_transform(x_test_100k.loc[:].values), columns=x_test_100k.columns)

    for samples in samples_list:
        perceptron_error_list = []
        perceptron_y_pred_list = []
        # model_sel_list = [] # uncomment if more than 32gb ram
        fold_selection = []
        dataset = generate_data(samples)
        if samples <= 200:
            batch_size = int(samples / 2)
        else:
            batch_size = 150
        # N-folds estimator
        if n_folds_crossvalidate:

            # Iterate through number of perceptrons in the first layer
            for num_perceptrons in range(1, perceptron_limit + 1):
                if not os.path.exists(OUTPATH + f'/{num_perceptrons}_perceptrons'):
                    os.makedirs(OUTPATH + f'/{num_perceptrons}_perceptrons')
                index_list = n_folds_calc(dataset, n_folds)
                acc_list = []
                acc_val_list = []
                loss_list = []
                loss_val_list = []
                epoch_list = []
                err_list = []
                y_pred_list = []
                # mdl_list = [] # uncomment if more than 32gb ram

                # Iterate through each fold
                for i in range(n_folds):
                    print(f'Folds {i}')
                    train, test = n_folds_split(dataset, index_list, i)

                    y_train = train[target_parameter]
                    y_test = test[target_parameter]

                    x_train = train.drop(target_parameter, axis=1)
                    x_train = pd.DataFrame(MinMaxScaler().fit_transform(x_train.loc[:].values), columns=x_train.columns)
                    x_test = test.drop(target_parameter, axis=1)
                    x_test = pd.DataFrame(MinMaxScaler().fit_transform(x_test.loc[:].values), columns=x_test.columns)
                    max_acc, max_acc_val, min_loss, min_loss_val, epoch_chosen, err, y_pred, mdl = simple_mlp_model(np.asarray(x_train),
                                                                                                                    y_train,
                                                                                                                    np.asarray(x_test),
                                                                                                                    y_test,
                                                                                                                    verbose=0,
                                                                                                                    layer_1_nodes=num_perceptrons,
                                                                                                                    optimal_train=optimal_train,
                                                                                                                    fold_number=i,
                                                                                                                    sample_number=samples,
                                                                                                                    save_figs=True,
                                                                                                                    batch_size=batch_size,
                                                                                                                    warm_epoch_count=epochs)
                    acc_list.append(max_acc)
                    acc_val_list.append(max_acc_val)
                    loss_list.append(min_loss)
                    loss_val_list.append(min_loss_val)
                    epoch_list.append(epoch_chosen)
                    err_list.append(err)
                    # mdl_list.append(mdl) # uncomment if more than system 32gb system ram

                index_chosen = err_list.index(min(err_list))
                perceptron_error_list.append(min(err_list))  # Append min error for given perceptron count across all n-folds
                # model_sel_list.append(mdl_list[index_chosen]) # uncomment if more than system 32gb ram
                fold_selection.append(index_chosen)  # Append the fold that generated the minimum error

            index_chosen = perceptron_error_list.index(min(perceptron_error_list))
            perceptron_chosen = index_chosen + 1
            print(f'Perceptron Count Chosen for Minimum Error: {perceptron_chosen}')

            use_fold = fold_selection[index_chosen]
            # model = model_sel_list[index_chosen] # uncomment if more than system 32gb ram
            # model_sel_list = [] # uncomment if more than system 32gb ram
            print(f'Fold Chosen: {use_fold}')
            train, test = n_folds_split(dataset, index_list, use_fold)

            y_train = train[target_parameter]
            y_test = test[target_parameter]

            x_train = train.drop(target_parameter, axis=1)
            x_train = pd.DataFrame(MinMaxScaler().fit_transform(x_train.loc[:].values), columns=x_train.columns)
            x_test = test.drop(target_parameter, axis=1)
            x_test = pd.DataFrame(MinMaxScaler().fit_transform(x_test.loc[:].values), columns=x_test.columns)
            verb = False

            _, _, _, _, _, _, _, model = simple_mlp_model(np.asarray(x_train),
                                                          y_train,
                                                          np.asarray(x_test),
                                                          y_test,
                                                          verbose=0,
                                                          layer_1_nodes=perceptron_chosen,
                                                          optimal_train=optimal_train,
                                                          fold_number=use_fold,
                                                          sample_number=samples,
                                                          save_figs=True,
                                                          batch_size=batch_size,
                                                          warm_epoch_count=epochs + 300)

            fig3 = plt.figure(3)
            plt.plot(acc_list)
            plt.plot(acc_val_list)
            plt.title(f'N-Folds Results (Accuracy) for {samples} Samples (Local Test Set)')
            plt.ylabel('Accuracy')
            plt.xlabel('Fold')
            plt.legend(['accuracy', 'validation accuracy'], loc='upper left')

            fig4 = plt.figure(4)
            plt.plot(loss_list)
            plt.plot(loss_val_list)
            plt.title(f'N-Folds Results (Loss) for {samples} Samples (Local Test Set)')
            plt.ylabel('Loss')
            plt.xlabel('Fold')
            plt.legend(['loss', 'validation loss'], loc='upper left')

            fig5 = plt.figure(5)
            plt.plot(perceptron_error_list)
            plt.axhline(y=opt_err, color='r', linestyle='-')
            plt.title(f'Perceptron Error vs. Optimal Error for {samples} Samples (Local Test Set)')
            plt.ylabel('Loss')
            plt.xlabel('Perceptrons')
            plt.legend(['Perceptron Loss', 'Optimal Loss'], loc='upper left')

            print(f'Running Final Evaluation on 100000 Test Set...')
            y_pred_100k = model.predict(x_test_100k)
            y_pred_100k = pd.DataFrame(y_pred_100k)
            y_pred_100k = y_pred_100k.idxmax(axis=1)
            fig7 = plt.figure(7)
            cm_100k = confusion_matrix(y_test_100k, y_pred_100k, normalize='true')
            plt.imshow(cm_100k, cmap='BuPu')
            for (i, j), label in np.ndenumerate(cm_100k):
                plt.text(j, i, str(round(label, 4)), ha='center', va='center')
            plt.colorbar
            error_100k = round(np.sum(1 - cm_100k.diagonal()) / cm_100k.shape[0], 4)
            plt.title(f'Confusion Matrix, Test Samples, {perceptron_chosen}')
            plt.ylabel('True Label')
            plt.xlabel(f'Predicted Label, Error = {error_100k}, Optimal Error = {opt_err}')

            fig8 = plt.figure(8, figsize=(12, 9))
            ax = fig8.add_subplot(projection='3d')
            plotting_df = pd.DataFrame(x_test_100k.values)
            plotting_df.columns = ['x', 'y', 'z']
            plotting_df['y_test'] = y_test_100k
            plotting_df['y_pred'] = y_pred_100k
            results_bool = y_pred_100k == y_test_100k
            plotting_df['results'] = results_bool.reset_index(drop=True).to_frame()

            marker_list = ['o', '<', 's', 'x', '*']

            for l in range(4):
                label_df = plotting_df[plotting_df['y_test'] == l]
                label_miss = label_df[label_df['results'] == False]
                label_match = label_df[label_df['results'] == True]
                ax.scatter(label_match['x'], label_match['y'], label_match['z'], color='g', marker = marker_list[l], label=f'Correct Class {l}')
                ax.scatter(label_miss['x'], label_miss['y'], label_miss['z'], color='r', marker=marker_list[l], label=f'Correct Class {l}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.title(f'{samples} Samples, {perceptron_chosen} Perceptrons, Fold {use_fold}, Predictions')
            plt.tight_layout()
            ax.legend()
            if verb:
                plt.show()

            fig3.savefig(OUTPATH + f'/results/{samples}_samples_{i}fold_acc_result')
            fig3.clear()
            fig4.savefig(OUTPATH + f'/results/{samples}_samples_{i}fold_loss_result')
            fig4.clear()
            fig5.savefig(OUTPATH + f'/results/{samples}_samples_{i}fold_error_result')
            fig5.clear()
            fig7.savefig(OUTPATH + f'/results/{samples}_samples_{perceptron_chosen}_perceptrons_{use_fold}_folds_test_result_confusion_matrix')
            fig7.clear()
            fig8.savefig(OUTPATH + f'/results/{samples}_samples_{perceptron_chosen}_perceptrons_{use_fold}_predictions')
            fig8.clear()

            index_chosen = perceptron_error_list.index(min(perceptron_error_list))
            sample_size_error_list.append(perceptron_error_list[index_chosen])
            sample_size_perceptrons.append(index_chosen + 1)

    fig6 = plt.figure(6)
    plt.semilogx(samples_list, sample_size_error_list, marker='.')
    plt.axhline(y=opt_err, color='r', linestyle='-')
    plt.title(f'Sample Size Error vs. Optimal Error')
    plt.ylabel('Loss')
    plt.xlabel('Sample Size')
    plt.legend(['Sample Loss', 'Optimal Loss'], loc='upper left')

    fig6.savefig(OUTPATH + f'/results/sample_size_error_vs_optimal_error')

    plt.show()
    print(f'Done...')
