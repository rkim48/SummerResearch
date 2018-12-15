from DCEpy.General.DataInterfacing.edfread import edfread
from DCEpy.Features.BurnsStudy.ictal_inhibitors_final import choose_best_channels
import os,sys, pickle
import numpy as np
import copy_reg, types

import matplotlib.pyplot as plt
from DCEpy.Features.Preprocessing.preprocessing import notch_filter_data


def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)


copy_reg.pickle(types.MethodType, _pickle_method)


def analyze_patient_raw_single_file(data_path, f_s, patient_id, type, ind):
    # ind is zero-indexed! If you want first interictal or ictal file, ind is 0!

    # specify data paths
    if not os.path.isdir(data_path):
        print data_path
        sys.exit('Error: Specified data path does not exist')

    # open the patient pickle file containing relevant information
    p_file = os.path.join(data_path, 'patient_pickle.txt')
    with open(p_file, 'r') as pickle_file:
        print("\tOpen Pickle: {}".format(p_file) + "...")
        patient_info = pickle.load(pickle_file)

    file_durations = patient_info['file_durations']
    print '\tFile durations of each edf file in seconds:', file_durations

    if type == 'ictal':
        try:
            data_filename = patient_info['seizure_data_filenames'][ind]
        except Exception, err:
            sys.exit('\tThe ictal file corresponding to index input does not exist! Please choose a smaller index!')
        ictal_filenames = list(patient_info['seizure_data_filenames'])
        seizure_times = patient_info['seizure_times'][ind]
        file_type = 'ictal'
        data_filename_path = os.path.join(data_path, data_filename)
        print '\tAvailable ictal files:', ictal_filenames

    if type == 'interictal':
        # get the awake interfiles and asleep interfiles (interictal files)
        interictal_chunk_num_dict = {}
        interictal_filenames = patient_info['awake_inter_filenames'] + patient_info['asleep_inter_filenames']
        for name in interictal_filenames:
            # create dictionary with keys as edf file name and values as length in 30 minute chunks
            interictal_chunk_num_dict[name] = int(np.ceil(file_durations[name] / 1800))

        total_list_length = 0
        edf_file_ind = 0
        interictal_chunk_num_list = interictal_chunk_num_dict.values()
        print '\tAvailable interictal files:', interictal_filenames

        while total_list_length <= ind:
            try:
                total_list_length += interictal_chunk_num_list[edf_file_ind]
            except Exception, err:
                sys.exit('\tThe interictal file corresponding to index input does not exist! Please choose a smaller index!')
            edf_file_ind += 1
        # print("edf_file_ind:", edf_file_ind - 1)
        # print("total_list_length:", total_list_length)
        desired_edf_file_index = edf_file_ind - 1
        desired_chunk_index = ind - total_list_length + interictal_chunk_num_list[desired_edf_file_index]
        # print("Desired chunk index:", desired_chunk_index)

        data_filename = interictal_filenames[desired_edf_file_index]
        seizure_times = None
        file_type = 'interictal'
        data_filename_path = os.path.join(data_path, data_filename)

    just_path, just_name = os.path.split(data_filename_path)
    print '\tThe chosen edf file name is:', just_name
    min_per_chunk = 30
    sec_per_min = 60

    print '\tGetting Data...'
    # for each relvant file for this  patient...
    all_files = []
    tmp_data_filenames = []
    tmp_file_type = []
    tmp_seizure_times = []

    if file_durations[just_name] >= min_per_chunk * sec_per_min and (file_type is 'interictal'):

        while True:
            # get chunk start and end times
            start = desired_chunk_index * sec_per_min * min_per_chunk
            end = (desired_chunk_index + 1) * sec_per_min * min_per_chunk
            print '\tStart time in seconds:', start
            print '\tEnd time in seconds:', end

            try:

                # extract the chunk
                print '\t\t\tChunk ' + str(ind) + ' reading...\n',
                # Gotta get the data_filename right!
                dimensions_to_keep = choose_best_channels(patient_id, seizure=0, filename=data_filename_path)
                X_chunk, _, labels = edfread(data_filename_path, rec_times=[start, end],
                                             good_channels=dimensions_to_keep)
                # Added: if the readed chunk is too short, break.
                if X_chunk.shape[0] < 300 * f_s:
                    print "Chunk is too short! Pick different chunk to analyze!"
                    break

                # update file information
                all_files.append(X_chunk)
                tmp_data_filenames = data_filename_path
                tmp_file_type = file_type
                tmp_seizure_times = seizure_times
                print '\t\t\tInterictal chunk %d reading complete!' % (ind)
                break

            except ValueError:
                print "\t\t\tFinished reading chunk"
                break
    else:
        print '\t\tIctal file %d reading...\n' % (ind),
        # read data in
        dimensions_to_keep = choose_best_channels(patient_id, file_type is 'ictal',
                                                  filename=data_filename_path)
        X, _, labels = edfread(data_filename_path, good_channels=dimensions_to_keep)

        print '\t\tStart time in seconds:', 0
        print '\t\tEnd time in seconds:', file_durations[just_name]
        print '\t\tIctal file %d reading complete!' % (ind)

        # update file information
        all_files.append(X)  # add raw data to files
        tmp_data_filenames = data_filename_path
        tmp_file_type = file_type
        tmp_seizure_times = seizure_times

    data_filenames = tmp_data_filenames
    file_type = tmp_file_type
    seizure_times = tmp_seizure_times

    # double check that no NaN values appear in the features
    for X, i in enumerate(all_files):
        if np.any(np.isnan(X)):
            print 'There are NaN in raw data of file', i
            sys.exit('Error: Uh-oh, NaN encountered while extracting features')

    return all_files, data_filenames, file_type, seizure_times

def read_single_file_corrected_index(data_path, f_s, patient_id, type, ind, channel):
    if type == 'ictal':
        all_files, _, _, seizuretimes = analyze_patient_raw_single_file(data_path, f_s, patient_id, type, ind)
    else:
        all_files, _, _, seizuretimes = analyze_patient_raw_single_file(data_path, f_s, patient_id, type, ind = ind-2)
    print all_files[0].shape
    return all_files[0][:,channel], seizuretimes

def grab_seizure_indices(data_path, ind):
    # specify data paths
    if not os.path.isdir(data_path):
        print data_path
        sys.exit('Error: Specified data path does not exist')

    # open the patient pickle file containing relevant information
    p_file = os.path.join(data_path, 'patient_pickle.txt')
    with open(p_file, 'r') as pickle_file:
        print("\tOpen Pickle: {}".format(p_file) + "...")
        patient_info = pickle.load(pickle_file)

    try:
        data_filename = patient_info['seizure_data_filenames'][ind]
    except Exception, err:
        sys.exit('\tThe ictal file corresponding to index input does not exist! Please choose a smaller index!')
    ictal_filenames = list(patient_info['seizure_data_filenames'])
    seizure_times = patient_info['seizure_times'][ind]
    print '\tSelected ictal file:', ictal_filenames[ind]
    print '\tAvailable ictal files:', ictal_filenames

    return seizure_times

# patient_id = 'TA023'
# # f_s = 1000
# to_data = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
# data_path = os.path.join(to_data, 'data')
# p_data_path = os.path.join(data_path, patient_id)
# # all_files, _, _, seizure_times = analyze_patient_raw_single_file(p_data_path, f_s, patient_id, 'ictal', 0)
# seizures = grab_seizure_indices(p_data_path, 3)
# print seizures
# # # print seizure_times_samples
# seizure_times_samples = np.multiply(seizure_times, f_s)
# plt.plot(np.array(all_files)[0, :, 0])
# print seizure_times_samples
# if seizure_times!= None:
#     plt.axvline(x = seizure_times_samples[0], label = 'seizure start', ls = "dashed")
#     plt.axvline(x = seizure_times_samples[1], label = 'seizure end', ls = "dashed")
# plt.show()

# channel_num = 5
# y = np.array(all_files)[0,:,channel_num]
# Y_nf = np.fft.fft(y)
# X_nofilter = y
# # [seizure_times_samples[0] - 100:seizure_times_samples[1] + 100]
# plt.subplot(2,3,1)
# plt.plot(X_nofilter)
# plt.title('No filter')
#
# y_filtered = notch_filter_data(y, 500) #filter the data <----- plot this in subplot 3
# N = len(y_filtered)
# Y_filtered = np.fft.fft(y_filtered)
# X = np.linspace(0, 500, N//2)
#
# Y_filtered_fft = np.fft.ifft(Y_filtered) # plot ifft of fft of filtered y
# # [seizure_times_samples[0] - 100:seizure_times_samples[1] + 100]
# plt.subplot(2,3,2)
# plt.plot(Y_filtered_fft)
# plt.title('IFFT of FFT of Filtered Signal')
#
# plt.subplot(2,3,3)
# plt.plot(y_filtered) # plot filtered y
# plt.title('Filtered Signal')
#
# plt.subplot(2,3,4)
# plt.plot(X, np.abs(Y_nf[:N//2]))
# plt.title('Non-filtered FFT of Channel {}'.format(channel_num))
# plt.ylabel('Amplitude')
# plt.yscale('log')
# plt.xlabel('Frequency (Hz)')
#
# plt.subplot(2,3,5)
# plt.plot(X, np.abs(Y_filtered[:N//2]))
# plt.title('Filtered FFT of Channel {}'.format(channel_num))
# plt.ylabel('Amplitude')
# plt.yscale('log')
# plt.xlabel('Frequency (Hz)')
# plt.show()
# plot seizure times



