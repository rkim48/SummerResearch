from DCEpy.General.DataInterfacing.edfread import edfread
from DCEpy.Features.BurnsStudy.ictal_inhibitors_final import choose_best_channels
import os,sys, pickle
import numpy as np
import matplotlib.pyplot as plt
from DCEpy.Features.Preprocessing.preprocessing import notch_filter_data

# Outputs only number of files, filenames, seizure times, and file types
# Does not output raw data itself and is thus useful for pipelines that utilize pre-calculated features
def fast_analyze_patient_raw(data_path):

    # specify data paths
    if not os.path.isdir(data_path):
        print "Data path:", data_path
        sys.exit('Error: Specified data path does not exist')

    # open the patient pickle file containing relevant information
    p_file = os.path.join(data_path, 'patient_pickle.txt')
    with open(p_file, 'r') as pickle_file:
        print("\tOpen Pickle: {}".format(p_file) + "...")
        patient_info = pickle.load(pickle_file)

    ictal_filenames = list(patient_info['seizure_data_filenames'])
    ictal_seizure_times = list(patient_info['seizure_times'])
    ictal_file_type = ['ictal'] * len(ictal_filenames)
    file_durations = patient_info['file_durations']
    print '\tFile durations of each edf file in seconds:', file_durations

    # get the awake interfiles and asleep interfiles (interictal files)
    interictal_chunk_num_dict = {}
    interictal_filenames = list(patient_info['awake_inter_filenames'] + patient_info['asleep_inter_filenames'])
    int_names = []
    interictal_files_length = 0
    for name in interictal_filenames:
        # create dictionary with keys as edf file name and values as length in 30 minute chunks
        interictal_chunk_num_dict[name] = int(np.ceil(file_durations[name] / 1800))
        int_names += [str(name)] * interictal_chunk_num_dict[name]
        interictal_files_length += int(np.ceil(file_durations[name] / 1800))

    interictal_seizure_times = [None] * interictal_files_length
    interictal_file_type = ['interictal'] * interictal_files_length

    number_of_files = interictal_files_length + len(ictal_filenames)
    seizure_times = ictal_seizure_times + interictal_seizure_times
    data_filenames = ictal_filenames + int_names
    file_type = ictal_file_type + interictal_file_type

    return number_of_files, data_filenames, file_type, seizure_times


# data_path = '\Users\Robin\Desktop\EpilepsyVIP\data\TA533'
# number_of_files, data_filenames, file_type, seizure_times = fast_analyze_patient_raw(data_path)
# print "Number of files:", number_of_files
# print "Data filenames:", data_filenames
# print "File types:", file_type
# print "Seizure times", seizure_times