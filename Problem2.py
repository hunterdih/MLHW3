import os
os.environ["OMP_NUM_THREADS"] = '1'
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import random
import math

random.seed(3)
OUTPATH = r'D:\MLResults\HW3\Problem2'
M1 = [5, 5]
C1 = [[0.2, 0],
      [0, 3]]
P1 = 0.25

M2 = [5, 5]
C2 = [[3, 0],
      [0, 0.2]]
P2 = 0.25

M3 = [5, 5]
C3 = [[0.5, 0],
      [0, 2]]
P3 = 0.25

M4 = [5, 5]
C4 = [[0.1, 0],
      [0, 4]]
P4 = 0.25


def generate_data(samples):
    data = []
    labels = []
    for i in range(samples):

        choice = random.random()
        if choice < P1:
            data.append(np.random.multivariate_normal(M1, C1, 1))
            labels.append(0)
        elif P1 <= choice < P1 + P2:
            data.append(np.random.multivariate_normal(M2, C2, 1))
            labels.append(1)
        elif P1 + P2 <= choice < P1 + P2 + P3:
            data.append(np.random.multivariate_normal(M3, C3, 1))
            labels.append(2)
        elif choice >= P1 + P2 + P3:
            data.append(np.random.multivariate_normal(M4, C4, 1))
            labels.append(3)
    data = np.stack(data, axis=1)
    data = pd.DataFrame(data[0])
    data['label'] = labels

    return data


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


def plot_gmm_results(data, predictions, gauss_comp, fld, test_number):
    df = pd.DataFrame(data)
    df.columns = ['x', 'y']
    df['y_pred'] = predictions
    classes = df['y_pred'].max()
    fig1 = plt.figure(1, figsize=(12, 9))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'{data.shape[0]} Samples, Fold {fld}, {gauss_comp} Gaussian Components')
    color_list = ['r', 'g', 'b', 'orange', 'brown', 'yellow']
    for i in range(classes + 1):
        temp_df = df[df['y_pred'] == i]
        plt.scatter(temp_df['x'], temp_df['y'], color=color_list[i])
    plt.legend(['1', '2', '3', '4', '5', '6', '7', '8'], loc='lower right')
    if not os.path.exists(OUTPATH + f'/{gauss_comp}_comp'):
        os.makedirs(OUTPATH + f'/{gauss_comp}_comp')
    fig1.savefig(OUTPATH + f'/{gauss_comp}_comp/{data.shape[0]}_samples_predictions_test_{test_number}')
    fig1.clear()


if __name__ == '__main__':
    n_fold_cross_validations = 10
    dataset_sizes = [10, 100, 1000, 10000]
    gmm_component_list = [1, 2, 3, 4, 5, 6]
    number_of_tests = 50
    gmm_component_selected_list = [0] * len(gmm_component_list)
    gmm_component_selected_per_sample = [0]*len(gmm_component_list)

    for test_number in range(number_of_tests):
        log_likelihood_per_sample = []
        best_model_per_sample = []
        M2[0]+=0.25
        M3[1]+=0.25
        M4[0]+=0.25
        M4[1]+=0.25

        for sample_size in dataset_sizes:
            dataset = generate_data(sample_size)
            dataset.columns = ['x', 'y', 'label']
            fig0 = plt.figure(0, figsize=(12,9))
            plt.scatter(dataset['x'], dataset['y'])
            if not os.path.exists(OUTPATH + '/results'):
                os.makedirs(OUTPATH + '/results')
            fig0.savefig(OUTPATH + f'/results/{sample_size}_sample_dataset_test_{test_number}')
            fig0.clear()
            fold_indexes = n_folds_calc(dataset, n_fold_cross_validations)

            best_model_per_component = []
            log_likelihood_per_component = []

            for gaussian_params in gmm_component_list:

                log_likelihood_fold_list = []

                for fold in range(n_fold_cross_validations):
                    pred_list = []
                    results_df = pd.DataFrame()

                    train, test = n_folds_split(dataset, fold_indexes, fold)
                    train_y = train['label']
                    train_x = train.drop('label', axis=1)
                    test_y = test['label']
                    test_x = test.drop('label', axis=1)

                    gmm = GaussianMixture(n_components=gaussian_params, n_init=5).fit(train_x)
                    pred_y = gmm.predict(test_x)

                    log_likelihood = gmm.score(test_x)
                    print(f'Log Likelihood For {test_number=}, {sample_size=}, {gaussian_params=}, {fold=} = {log_likelihood}')
                    log_likelihood_fold_list.append(log_likelihood)

                index = log_likelihood_fold_list.index(max(log_likelihood_fold_list))
                log_likelihood_per_component.append(log_likelihood_fold_list[index])
                best_model_per_component.append(index + 1)

                train, test = n_folds_split(dataset, fold_indexes, index)
                train_y = train['label']
                train_x = train.drop('label', axis=1)
                test_y = test['label']
                test_x = test.drop('label', axis=1)
                gmm = GaussianMixture(n_components=gaussian_params, n_init=5).fit(train_x)
                pred_y = gmm.predict(test_x)
                plot_gmm_results(test_x, pred_y, gaussian_params, index + 1, test_number)

            index = log_likelihood_per_component.index(max(log_likelihood_per_component))
            max_log_likelihood = log_likelihood_per_component[index]
            best_model = best_model_per_component[index]
            log_likelihood_per_sample.append(max_log_likelihood)
            best_model_per_sample.append(index + 1)
            gmm_component_selected_per_sample[index] += 1

            fig2 = plt.figure(2, figsize=(12,9))
            plt.title('Log Likelihood vs. Gaussian Parameters')
            plt.plot(gmm_component_list, log_likelihood_per_component)
            plt.ylabel('Log Likelihood')
            plt.xlabel('Gaussian Parameters')
            plt.xticks(gmm_component_list)
            fig2.savefig(OUTPATH + f'/results/{sample_size}_sample_log_likelihood_param_test_{test_number}')
            fig2.clear()

        fig3 = plt.figure(3, figsize=(12,9))
        plt.title('Log Likelihood vs. Sample Size')
        plt.ylabel('Log Likelihood')
        plt.xlabel('Sample Size')
        plt.semilogx(dataset_sizes, log_likelihood_per_sample, marker='.')
        fig3.savefig(OUTPATH + f'/results/log_likelihood_vs_sample_size_test_{test_number}')
        fig3.clear()

        index = log_likelihood_per_sample.index(max(log_likelihood_per_sample))
        gmm_component_selected_list[index] += 1

    fig4 = plt.figure(4, figsize=(12,9))
    plt.title('GMM Component Selection Occurrences (Max Likelihood Per Samples)')
    plt.ylabel('Occurrences')
    plt.xlabel(f'GMM Model, (Values={gmm_component_selected_list}')
    plt.xticks(gmm_component_list)
    plt.bar(gmm_component_list, gmm_component_selected_list)
    fig4.savefig(OUTPATH + f'/results/GMM_OCCURRENCES')
    fig4.clear()

    fig5 = plt.figure(5, figsize=(12,9))
    plt.title('GMM Component Selection Occurrences\n(Max Likelihood Each Sample, Each Test)')
    plt.ylabel('Occurrences')
    plt.xlabel(f'GMM Model, (Values={gmm_component_selected_per_sample}')
    plt.xticks(gmm_component_list)
    plt.bar(gmm_component_list, gmm_component_selected_per_sample)
    fig5.savefig(OUTPATH + f'/results/GMM_OCCURRENCES_PER_SAMPLE_PER_TEST')
    fig5.clear()
