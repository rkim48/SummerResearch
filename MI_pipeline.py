import pickle
from scipy import io
from DCEpy.Features.BurnsStudy.Mutual_Information.centralities import *
from sklearn import svm
import os, sys
from email_results import email_results
# to_prepr = "/home/rk48/Desktop/EpilepsyVIP/DCEpy/Features/Preprocessing/"
# to_edf = "/home/rk48/Desktop/EpilepsyVIP/DCEpy/General/DataInterfacing/"
# to_burns = "/home/rk48/Desktop/EpilepsyVIP/DCEpy/Features/BurnsStudy"
# sys.path.append(to_prepr)
# sys.path.append(to_edf)
# sys.path.append(to_burns)
#
# os.chdir(to_edf)
# cwd = os.getcwd()
# os.chdir(to_burns)

# os.chdir(cwd)
# from ictal_inhibitors_final import classifier_gridsearch, classifier_gridsearch_IF, label_classes, window_to_samples, choose_best_channels, performance_stats
# from fast_analyze_patient_raw import fast_analyze_patient_raw


def get_freq_bands():
    return ["delta", "theta", "alpha", "beta"], [[0.5,4], [4,8], [8,13], [13, 35]]

def update_log(log_file, prediction_sensitivity, detection_sensitivity, latency, fp, time, freqband, centrality_type):

    # records results of features with different hyperparameters such as frequency bands and centrality type
    f = open(log_file, 'a')

    if (np.nansum(fp) / np.nansum(time)) < 1.0 and np.nanmean(prediction_sensitivity) == np.nanmean(detection_sensitivity) == 1.0:
        f.write('!!!!! GOOD RESULTS FOUND !!!!!' + '\n')
    f.write('Frequency Bands: ' + '\t' + str(freqband) + '\n')
    f.write('Centrality Type: ' + '\t' + centrality_type + '\n')
    f.write('Mean Prediction Sensitivity: \t%.2f\n' % (np.nanmean(prediction_sensitivity)))
    f.write('Mean Detection Sensitivity: \t%.2f\n' % (np.nanmean(detection_sensitivity)))
    f.write('Mean Latency: \t%.4f\n' % (np.nanmean(latency)))
    f.write('False Positive Rate: \t%.5f (fp/Hr) \n\n' % (np.nansum(fp) / np.nansum(time)))

    f.close()
 
    return

def get_seizure_time(patient_id, filename):
    home_path = "/Users/Robin/Desktop/EpilepsyVIP/data"
    data_path = os.path.join(home_path, patient_id)

    # specify data paths
    if not os.path.isdir(data_path):
        sys.exit('Error: Specified data path does not exist')

    # open the patient pickle file containing relevant information
    p_file = os.path.join(data_path, 'patient_pickle.txt')
    with open(p_file, 'r') as pickle_file:
        print("\tOpen Pickle: {}".format(p_file) + "...")
        patient_info = pickle.load(pickle_file)

    # add data file names, seizure times, file types
    data_filenames = list(patient_info['seizure_data_filenames'])
    seizure_times = list(patient_info['seizure_times'])

    return seizure_times[data_filenames.index(filename)]

def get_seizure_windows(seizure_time, win_len, win_overlap):
    # determine seizure start/end times in seconds
    seizure_start_time = seizure_time[0]
    seizure_end_time = seizure_time[1]

    # determining which window the seizure starts in
    if seizure_start_time < win_len:
        seizure_start_window = 0
    else:
        seizure_start_window = int((seizure_start_time - win_len) / (win_len - win_overlap) + 1)

    # determining which window the seizure ends in
    if seizure_end_time < win_len:
        seizure_end_window = 0
    else:
        seizure_end_window = int((seizure_end_time - win_len) / (win_len - win_overlap) + 1)

    return seizure_start_window, seizure_end_window

def label_classes(num_windows, preictal_time, postictal_time, win_len, win_overlap, seizure_time, file_type):
    # labeling the seizure files
    if file_type is 'ictal':

        labels = np.empty(num_windows)

        # determine seizure start/end times in seconds
        seizure_start_time = seizure_time[0]
        seizure_end_time = seizure_time[1]

        # determining which window the seizure starts in
        if seizure_start_time < win_len:
            seizure_start_window = 0
        else:
            seizure_start_window = int((seizure_start_time - win_len) / (win_len - win_overlap) + 1)

        # determining which window the seizure ends in
        if seizure_end_time < win_len:
            seizure_end_window = 0
        else:
            seizure_end_window = int((seizure_end_time - win_len) / (win_len - win_overlap) + 1)

        # in case the seizure end window is larger than the max window index
        if seizure_end_window > num_windows - 1:
            seizure_end_window = num_windows - 1

        # label the ictal period
        labels[seizure_start_window:seizure_end_window + 1] = -np.ones(seizure_end_window + 1 - seizure_start_window)

        # label the preictal (and interictal period if that exists) period
        if seizure_start_time > preictal_time + win_len:  # if there is a long period before seizure onset

            # determine the time in seconds where preictal period begins
            preictal_start_time = seizure_start_time - preictal_time

            # determine the time in windows where preictal period begins
            preictal_start_win = int((preictal_start_time - win_len) / (win_len - win_overlap) + 1)

            # label the preictal time
            labels[preictal_start_win:seizure_start_window] = -np.ones(seizure_start_window - preictal_start_win)

            # label the interical time
            labels[:preictal_start_win] = np.ones(preictal_start_win)

        else:  # if there is not a long time in file before seizure begins
            # label preictal time
            labels[:seizure_start_window] = -np.ones(seizure_start_window)

        # determining how long the postical period lasts in seconds
        postictal_period = (num_windows - seizure_end_window) * (win_len - win_overlap)

        # if there is a long period of time after seizure in the file
        if postictal_period > postictal_time:

            # determine where in seconds the postical period ends
            postictal_end_time = seizure_end_time + postictal_time

            # determine where in windows the postical period ends
            postictal_end_win = int((postictal_end_time - win_len) / (win_len - win_overlap) + 1)

            # in the case that the postictal end window exceeds the maximum number of windows...
            if postictal_end_win > num_windows - 1:
                postictal_end_win = num_windows - 1

            # label the postical period
            labels[seizure_end_window + 1:postictal_end_win + 1] = -np.ones(postictal_end_win - seizure_end_window)

            # label the interictal period
            labels[postictal_end_win + 1:] = np.ones(num_windows - 1 - postictal_end_win)

        else:  # if there is a short amount of time in the file after the seizure ends
            # label the postictal period
            labels[seizure_end_window + 1:] = -np.ones(num_windows - 1 - seizure_end_window)

    # label awake interictal files
    elif file_type is 'interictal':
        labels = np.ones(num_windows)

    # label asleep interictal files
    elif file_type is 'interictal':
        labels = np.ones(num_windows)

    # return the data labels
    return list(labels)

def find_centrality_multibands(training_MI_files, centrality_type):
    # input: list of (n_samples for this file, num_freq, num_channels, num_channels)
    # output: list of (n_samples for this file, num_freq, num_channels)
    training_centrality_files = []

    for file in training_MI_files:
        n_samples, n_freq, n_channels, _ = file.shape
        interictal_centrality = np.zeros((n_samples, n_freq, n_channels))
        for i in range(n_samples):
            for j in range(n_freq):
                # TODO: experiment with different centrality values. functions are imported from centralities.py. find_centralities_multiband only supports the centrality functions not the stats one.
                if centrality_type == 'katz':
                    interictal_centrality[i, j, :] = compute_katz(file[i, j, :, :])
                elif centrality_type == 'eig':
                    interictal_centrality[i, j, :] = eigen(file[i, j, :, :])
                elif centrality_type == 'pagerank':
                    interictal_centrality[i, j, :] = pagerank_centrality(file[i, j, :, :])

        r_interictal_centrality = np.reshape(interictal_centrality, (n_samples, n_freq * n_channels))
        training_centrality_files.append(r_interictal_centrality)

    return training_centrality_files

def get_MI_features(patient_id, filename, feature_set, freqbands):

    # TODO: change this to whatever directory you store the raw patient data
    h_path = '/media/ExtHDD2/rk48Ext/data'
    f_path = os.path.join('/home/rk48/Desktop/EpilepsyVIP/DCEpy/Features/BurnsStudy/Mutual_Information', patient_id)

    # TODO: modify the directories based on where you place the Epilepsy VIP folder
    data_mat = scipy.io.loadmat(os.path.join(f_path, "MI_{}.mat".format(feature_set)))[filename]

    all_band_names, bands = get_freq_bands()

    for i, band_name in enumerate(freqbands):
        band_index = all_band_names.index(band_name)
        if i == 0:
            MI = data_mat[:, [band_index], :, :]
        else:
            MI = np.hstack((MI, data_mat[:, [band_index], :, :]))
    return np.array(MI)

def find_normalizing_MI(matrix):
    # compute the mean of each entry along the third dimension
    mean_mat = np.mean(matrix, axis= 0)

    # compute the standard deviation of each entry along the third dimension
    std_mat = np.std(matrix, axis= 0)

    return mean_mat, std_mat


def transform_coherency(coherencies_list, mean, std):
    # fail-safe to avoid dividing by zero
    std[std == 0] = 1
    transformed_coherencies = []

    # for each file's coherency matrices...
    for coherency_matrices_one_file in coherencies_list:

        # for each window's coherency matrix...
        num_windows = coherency_matrices_one_file.shape[0]

        for i in xrange(num_windows):
            matrix = coherency_matrices_one_file[i, :,  :, :].copy()

            # normalize the matrix. This is done according to Burns et. al.
            matrix -= mean
            # matrix = np.divide(matrix, std)
            # matrix = np.divide(np.exp(matrix), 1 + np.exp(matrix))

            # store all transformed coherence matrices for this file
            coherency_matrices_one_file[i, :, :, :] = matrix

        # store all transformed coherence matrices for all files
        transformed_coherencies += [coherency_matrices_one_file]

    return transformed_coherencies


def test_transform_MI(coherency_matrix, mean, std, num_channels):
    # step through each value in the coherence matrix
    band_num = coherency_matrix.shape[0]
    for band in range(band_num):
	    for row in np.arange(num_channels):
	        for col in np.arange(num_channels):
	            # normalize
	            coherency_matrix[band, row, col] -= mean[band, row, col]
	            # coherency_matrix[band, row, col] = float(coherency_matrix[band, row, col]) / float(std[band, row, col])
	            # # transform
	            # coherency_matrix[band, row, col] = (2.7182818284590452353602874713527 ** coherency_matrix[band, row, col])
	            # denominator = coherency_matrix[band, row, col] + 1
	            # coherency_matrix[band, row, col] = coherency_matrix[band, row, col] / float(denominator)
    return coherency_matrix


def offline_training(cv_file_type, cv_file_names, feature_set, cv_file_idxs, cv_seizure_times, chunk_len, chunk_overlap,
                     patient_id, preictal_time, postictal_time, svm_kernel = 'rbf'):

    calc_features_local = 0
    first_file = 0

    # read pre-calculated features
    if not calc_features_local:
        print'\t\tBuilding MI matrices for all training files'
        training_MI_cv_files = []
        for n in xrange(len(cv_file_type)):
            # fetch data
            filename = cv_file_names[n]
            key = str(cv_file_idxs[n]) + "_" + filename.split("\\")[-1]
            test_file_MI = get_MI_features(patient_id, key, feature_set)    #(111, 2, 6, 6)

            training_MI_cv_files += [test_file_MI]

            # store specifically interictal MI matrices
            if cv_file_type[n] is not "ictal":
                if first_file == 0:
                    # interictal files are for finding normalizing parameters
                    interictal_MI_files = test_file_MI     # should be all number of samples, 2 * 6 * 6
                    first_file = 1
                else:
                    interictal_MI_files = np.vstack((interictal_MI_files,
                                                      test_file_MI))

    # After getting okay results: implement local feature calculation
    else:
        raise ValueError("Uh-oh, local feature calculation not implemented yet.")
    print interictal_MI_files.shape

    print '\t\tFinding mean and standard deviation of interictal features'
    mean_MI_matrix, sd_MI_matrix = find_normalizing_MI(interictal_MI_files)

    print'\t\tTransforming all coherency matrices'
    transformed_MI_cv_files = transform_coherency(training_MI_cv_files, mean_MI_matrix,
                                                         sd_MI_matrix)


    training_katz_cv_files = find_centrality_multibands(transformed_MI_cv_files)  # should be list of (n_samples, 2, 6, 6)

    # initializations
    training_data = training_katz_cv_files
    interictal_indices = []
    seizure_indices = []

    # stack all interictal data
    interictal_data = np.vstack(
        (training_data[ind] for ind in xrange(len(training_data)) if cv_file_type[ind] is not 'ictal'))
    print interictal_data.shape
    # organizing the data and labeling it
    # create label dictionary to store labels of CV training files
    cv_label_dict = {}
    # label each training file
    for index in xrange(len(cv_file_type)):

        # if file is seizure, store this training file index to seizure indices
        if cv_file_type[index] == "ictal":
            seizure_indices.append(index)

        # otherwise, store this training file index to interictal indices
        else:
            interictal_indices.append(index)

        # label the windows in the file
        cv_label_dict[index] = label_classes(len(training_data[index]), preictal_time, postictal_time, chunk_len,
                                             chunk_overlap, cv_seizure_times[index], cv_file_type[index])

    print '\t\tTraining the classifier'
    # tuning the SVM parameterss
    bestnu, bestgamma = classifier_gridsearch(training_data, cv_label_dict, seizure_indices, interictal_indices, svm_kernel)

    # define your classifier
    best_clf = svm.OneClassSVM(nu=bestnu, kernel=svm_kernel, gamma=bestgamma)

    best_clf.fit(interictal_data)


    return best_clf, mean_MI_matrix, sd_MI_matrix


def online_testing(feature,pred_time,label_index,alarm_timer,best_clf):

    testing_label = best_clf.predict(feature)

    # switch the labels to our code's convention: 0 is normal, 1 is seizure
    testing_label[testing_label == 1] = 0
    testing_label[testing_label == -1] = 1


    # determining where the outlier fraction meets or exceeds the threshold
    if alarm_timer <= 0:
        if testing_label == 1:
            decision = 1  # predicted a seizure
            alarm_timer = pred_time  # set the refractory period
        else:
            decision = -1  # did not predict a seizure

    else:  # do not output a positive prediction within an alarm_timer period of time of the last positive prediction
        decision = -1
        alarm_timer -= 1

    return decision, label_index, alarm_timer

def feature_pipeline(feature_set, win_len, freq_bands, centrality_type):
    # TODO: set patients
    patients = ['TA533']  # set which patients you want to test

    # save path directory for log file where results will be stored
    save_path = '/home/rk48/Desktop/EpilepsyVIP/DCEpy/Features/BurnsStudy/Mutual_Information'

    # setting model parameters
    chunk_len = win_len
    chunk_overlap = int(chunk_len * 0.75)
    f_s = float(1e3)  # Hz

    persistence_time = 300/(chunk_len - chunk_overlap) + 1
    # minutes times seconds, the amount of time after a seizure prediction for which no alarm is raised
    preictal_time = 5 * 60  # minutes times seconds, the amount of time prior to seizure onset defined as preictal
    postictal_time = 5 * 60  # minutes times seconds, the amount of time after seizure end defined as postictal

    # TODO: set rbf kernel here.
    svm_kernel = 'rbf'

    # evaluate each patient
    for patient_index, patient_id in enumerate(patients):

        print "\n---------------------------Analyzing patient ", patient_id, "----------------------------\n"

        # update paths to be specific to each patient
        my_data_path = '/media/ExtHDD2/rk48Ext/data'
        p_data_path = os.path.join(my_data_path, patient_id)


        print 'Retreiving stored raw data'
        number_files, data_filenames, file_type, seizure_times = fast_analyze_patient_raw(p_data_path)

        # intializing performance stats
        prediction_sensitivity = np.zeros(number_files)
        detection_sensitivity = np.zeros(number_files)
        latency = np.zeros(number_files)
        fp = np.zeros(number_files)
        times = np.zeros(number_files)
        # beginning leave one out cross-validation
        for i in xrange(number_files):

            print '\nCross validations, k-fold %d of %d ...' % (i + 1, number_files)
            # defining which files are training files vs testing files for this fold
            testing_file_idx = i
            cv_file_names = data_filenames[:i] + data_filenames[i + 1:]
            cv_file_idxs = range(i) + range(i+1,number_files)

            cv_file_type = file_type[:i] + file_type[i + 1:]
            cv_seizure_times = seizure_times[:i] + seizure_times[i + 1:]

            print '\tEntering offline training'
            my_svm, mean_MI_matrix, sd_MI_matrix = offline_training(cv_file_type, cv_file_names, feature_set, cv_file_idxs, cv_seizure_times, chunk_len, chunk_overlap, patient_id,
                                      preictal_time, postictal_time, svm_kernel, freq_bands)


            print'\tEntering online testing'
            # computing prediction on testing file for this fold

            print '\tCalculating testing features locally'

            # load test file
            test_key = str(testing_file_idx) + "_" + data_filenames[testing_file_idx].split("\\")[-1]
            test_MI = get_MI_features(patient_id, test_key, feature_set, freq_bands)

            # transform (normalize) MI matrix
            transformed_MI_test = transform_coherency([test_MI], mean_MI_matrix,
                                                          sd_MI_matrix)

            test_features = find_centrality_multibands(transformed_MI_test, centrality_type)[0]

            # should be list of (n_samples, 2, 6, 6)  # for loop to process each window in the test file
            print "test features", test_features.shape

            t_samples = test_features.shape[0]
            full_file_decision = np.zeros(t_samples)

            alarm_timer = 0

            for index in np.arange(t_samples):
                # getting the single window of data for this iteration of the for loop
                feature = test_features[index].reshape(1, -1)
                print test_features[index].shape
                print "Feature shape:", feature.shape
                decision, label_index, alarm_timer = online_testing(feature, persistence_time, index,
                                                                    alarm_timer, my_svm)

                # storing the outlier fraction and decision for calculating performance metrics and visualization
                full_file_decision[index] = decision

            # using outputs from test file to compute performance metrics
            print'\tCalculating performance stats'

            print "\tFile Type: ", file_type[i]
            print "\tFile name: ", data_filenames[i]

            print "\tFull File Decision: ", full_file_decision

            # convert from units of windows to units of samples
            test_decision_sample = window_to_samples(full_file_decision, chunk_len, chunk_overlap, f_s)

            # find performance metrics for this fold of cross validation
            prediction_sensitivity[i], detection_sensitivity[i], latency[i], fp[i], times[i] = performance_stats(
                test_decision_sample, seizure_times[i], f_s, preictal_time, chunk_len, chunk_overlap)


            # print the performance metrics and visualize the algorithm output on a graph
            print '\tPrediction sensitivity = ', prediction_sensitivity[i], 'Detection sensitivity = ', \
            detection_sensitivity[i], 'Latency = ', latency[i], 'FP = ', fp[i], 'Time = ', times[i]

        # compute false positive rate
        fpr = float(np.nansum(fp)) / float(np.nansum(times))

        # print mean and median performance metrics
        print '\nMean prediction sensitivity = ', np.nanmean(
            prediction_sensitivity), 'Mean detection sensitivity = ', np.nanmean(
            detection_sensitivity), 'Mean latency = ', np.nanmean(latency), 'Mean FPR = ', fpr
        print 'Median prediction sensitivity = ', np.nanmedian(
            prediction_sensitivity), 'Median detection sensitivity = ', np.nanmedian(
            detection_sensitivity), 'Median latency = ', np.nanmedian(latency)

        log_file = os.path.join(save_path, 'log_file.txt')
        update_log(log_file, prediction_sensitivity, detection_sensitivity, latency, fp, times, freq_bands, centrality_type)



