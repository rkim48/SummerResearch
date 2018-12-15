from scipy import special, signal
import os, sys
# to_prepr = "/home/rk48/Desktop/EpilepsyVIP/DCEpy/Features/Preprocessing/"
# to_edf = "/home/rk48/Desktop/EpilepsyVIP/DCEpy/General/DataInterfacing/"
# to_burns = "/home/rk48/Desktop/EpilepsyVIP/DCEpy/Features/BurnsStudy"
# sys.path.append(to_prepr)
# sys.path.append(to_edf)
# sys.path.append(to_burns)

from preprocessing import notch_filter_data, get_freq_bands
# os.chdir(to_edf)
# cwd = os.getcwd()
from edfread import edfread
# os.chdir(to_burns)
from DCEpy.Features.BurnsStudy.ictal_inhibitors_final import analyze_patient_raw
from centralities import *
import numpy.random as nr
import scipy.spatial as spatial
import scipy.io as sio
import scipy.signal
import numpy as np
from read_single_file import analyze_patient_raw_single_file
from permutation_entropy import compute_MI
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# os.chdir(cwd)
import pickle
import scipy
import math

def mutual_information_ksg2004_XY(x, y, k = 3, seed = 1, threshold = True ):
    """" Mutual information between x and y in nats. Implementation by Rakesh. """
    # tol - small noise to break degeneracy, see doc.
    tol = 1e-10
    assert len(x) == len(y)
    assert k <= len(x) - 1  # to ensure k^th nearest neighbor exists for each point in the x and y

    nr.seed(seed = seed)
    x = [list(p + tol * nr.rand(len(x[0]))) for p in x]
    y = [list(p + tol * nr.rand(len(y[0]))) for p in y]

    data = [list(x[ii] + y[ii]) for ii in range(len(x))]

    # Build a kd-tree data structure for the data in joint space and in marginal subspaces, to
    # ensure faster k-nn queries
    xy_tree = spatial.cKDTree(data)
    x_tree = spatial.cKDTree(x)
    y_tree = spatial.cKDTree(y)
    # Compute the distance between each data point and its kth nearest neighbor
    xy_dist = [xy_tree.query(elem, k=k +1, p=float('inf'))[0][k] for elem in data]
    x_num = [len(x_tree.query_ball_point(x[ii], xy_dist[ii] - 1e-15, p=float('inf'))) for ii in
             range(len(x))]
    y_num = [len(y_tree.query_ball_point(y[ii], xy_dist[ii] - 1e-15, p=float('inf'))) for ii in
             range(len(y))]

    mi_est = special.digamma(k) + special.digamma(len(x)) - (sum(special.digamma( x_num))
                                                             + sum(special.digamma( y_num))) / len(x)

    if threshold:
        if mi_est < 0:
            mi_est = 0

    return mi_est

def PMI(x, y):
    real_x = x[:,0]
    real_y = y[:,0]
    imag_x = x[:,1]
    imag_y = y[:,1]
    # return the average of MI between Re(x) and Re(y) AND Im(x) and Im(y)
    # plt.figure()
    # plt.subplot(2,2,1)
    # plt.title("real x")
    # plt.plot(real_x)
    # plt.subplot(2,2,2)
    # plt.title("real y")
    # plt.plot(real_y)
    # plt.subplot(2,2,3)
    # plt.plot(imag_x)
    # plt.subplot(2,2,4)
    # plt.plot(imag_y)
    # plt.show()
    print "MI real:", compute_MI(real_x,real_y,m=3,delay=1)
    print "MI imag:", compute_MI(imag_x,imag_y,m=3,delay=1)
    PMI = (compute_MI(real_x,real_y,m=3,delay=1) + compute_MI(imag_x,imag_y,m=3,delay=1)) / 2.0
    print "PMI:", PMI

    return PMI


def mi_in_frequency_channel(dXband, dYband, freqband):
    ndf = dXband.shape[1]  # ndf is length of dXband, specific frequency band of the DFT
    mi_band = []
    for i in range(ndf):
        real_dXi = np.hstack((np.real(dXband[:, [i]]), np.imag(dXband[:, [i]])))
        real_dYi = np.hstack((np.real(dYband[:, [i]]), np.imag(dYband[:, [i]])))
        # mi_band.append(mutual_information((real_dXi, real_dYi), k=3))
        mi_band.append(mutual_information_ksg2004_XY(real_dXi, real_dYi, k = 3))
    return np.array(mi_band)


def cross_channel_mutual_information_in_frequency(X, Y, resolution = 5, freqbands = [[1, 100]],  f_s = 1000):
    """
    Computes mutual information between two signals for the selected frequency bands.
    X: NxNs matrix, where N is the number of windows per chunk and Ns is the length of a window.
       each signal can be from a single channel or the average of all channels.
    Y:  NxNs matrix, where N is the number of windows per chunk and Ns is the length of a window.
       each signal can be from a single channel or the average of all channels.
    Nf: DFT length.
    fhigh: max frequency. Should be less than 500Hz.
    """

    Nf = f_s / resolution
    dX = np.fft.fft(X, n=Nf, axis=1, norm='ortho')
    length = dX.shape[1]
    dY = np.fft.fft(Y, n=Nf, axis=1)
    frequencies = np.fft.fftfreq(Nf) * f_s
    freq_length = frequencies.shape[0]
    print "Shape of X:", X.shape

    # plt.plot(frequencies[:length//2], abs(dX[0,:length//2]))
    # plt.show()
    decimate_number = 2
    resolution_2 = 5
    fs_2 = f_s / decimate_number
    Nf2 = fs_2/resolution_2
    assert(Nf2 <= length / decimate_number)
    frequencies_2 = np.fft.fftfreq(Nf2) * fs_2
    #
    #
    X_ds = scipy.signal.decimate(X, decimate_number, axis=1)

    print "Shape of downsampled X:", X_ds.shape
    dX_ds = np.fft.fft(X_ds, n=Nf2, axis=1, norm="ortho")
    length2 = dX_ds.shape[1]
    print "Shape of dX_ds:", dX_ds.shape


    plt.subplot(2, 2, 1)
    plt.title("Original Signal")
    plt.plot(X[0,:])
    plt.subplot(2, 2, 2)
    plt.title("Downsampled Signal")
    plt.plot(X_ds[0,:])
    plt.subplot(2, 2, 3)
    plt.title("FFT of Original Signal")
    print "frequencies:", frequencies.shape
    print "frequencies_2:", frequencies_2.shape
    plt.stem(frequencies[:length//2], abs(dX[0,:length//2]))
    plt.subplot(2, 2, 4)
    plt.title("FFT of Downsampled Signal")
    plt.stem(frequencies_2[:length2//2], abs(dX_ds[0,:length2//2]))
    plt.show()
    print "Shape of dX", dX.shape




    list_mi_bands = []
    print freqbands
    # plt.figure()
    for freqband in freqbands:
        # select the FFT within a frequency band
        freq_indices = np.where((freqband[0] <= frequencies) & (frequencies <= freqband[1]))[0]
        dXband = dX[:, freq_indices]
        dYband = dY[:, freq_indices]
        print "dXband", dXband.shape
        print "dYband", dYband.shape

        # compute cross channel MI in frequency
        mi_band = mi_in_frequency_channel(dXband, dYband, freqband)

        list_mi_bands.append(np.mean(mi_band))

    print "MI of the band: ", list_mi_bands
    return list_mi_bands


def cross_channel_MI(X_chunk, freqbands = [[1, 100]], Nf = 500, f_s = 1000):
    """
    Returns cross channel mutual information for all frequency bands. Should be
    (number of channels, number of channels, number of frequency bands)
    """
    # for every two channels
    N, Ns, p = X_chunk.shape
    print "YEET:",X_chunk.shape
    cmi_graph = np.zeros(shape=(p, p, len(freqbands)))

    for i in xrange(p):
        for j in xrange(i, p):
            i = 0
            j = 1
            X_channel_i = X_chunk[:, :, i]
            X_channel_j = X_chunk[:, :, j]
            if i == j:
                cmi_graph[i, j, :] = np.zeros(shape=(len(freqbands),))
            else:
                cmi_graph[i, j, :] = cross_channel_mutual_information_in_frequency(X_channel_i, X_channel_j, freqbands=freqbands)
                cmi_graph[j, i, :] = cmi_graph[i, j, :]

    list_mis_allbands = []
    for i in range(len(freqbands)):
        list_mis_allbands.append(cmi_graph[:,:,i])

    # shape should be number of (frequencies, number of channels, number of channels)
    return np.array(list_mis_allbands)


def window_data(data, win_len, win_ovlap, f_s = 1000):
    n = data.shape[0]
    all_windowed_data = []

    # getting window information in units of samples
    win_len = win_len * f_s
    win_ovlap = win_ovlap * f_s

    # computing the number of windows from the given data in units of samples
    num_windows = int(math.floor(float(n) / float(win_len - win_ovlap)))

    # compute the coherency matrix for each window of data in the file
    for index in np.arange(num_windows):
        # find start/end indices for window within file
        start = index * (win_len - win_ovlap)
        end = min(start + win_len, n)

        # get window of data
        if end <= n:
            window_of_data = data[int(start):int(end), :]     # multidimensional
            # print "windowed data shape : ", window_of_data.shape
            if window_of_data.shape[0] < win_len:                      # get rid of short windows in the end
                break
            all_windowed_data.append(window_of_data)
    return np.array(all_windowed_data)

def extract_CMI_whole_file(X, filename, patient_id, i, freqbands = [[1, 100]],  save_file = False, chunk_len = 180, chunk_ovlp = 90, mi_win_len = 1, mi_win_overlap = 0.5):
    """
    If you want to extract MI features for a file, use this one.

    """
    mi_list = []

    # window data
    chunked_data = window_data(X, win_len= chunk_len, win_ovlap= chunk_ovlp)
    print "original data shape: ", X.shape
    print "chunked data shape: ", chunked_data.shape

    # compute MI information for each chunk
    for chunk_idx in range(0, chunked_data.shape[0]):
        chunk = chunked_data[chunk_idx, :, :]
        windowed_chunk =  window_data(chunk, win_len = mi_win_len, win_ovlap = mi_win_overlap)
        print "windowed chunk shape: ", windowed_chunk.shape
        # compute cross channel
        mi_all_freqs = cross_channel_MI(windowed_chunk, freqbands=freqbands , Nf = 500, f_s = 1000)
        mi_list.append(mi_all_freqs)

    print "\t MI matrix shape (should be 6*6*6): ", mi_all_freqs.shape

    if save_file == True:
        index = filename.rfind('/')
        truncated_filename = filename[index + 1:]
        if i == 0:
            # scipy.io.savemat(patient_id + "corrected_CMI.mat", {str(i) + "_" + truncated_filename: mi_list})
            scipy.io.savemat(patient_id + "CMI_5m_new.mat", {str(i) + "_" + truncated_filename: mi_list})

        else:
            # mat_dict = scipy.io.loadmat(patient_id + "corrected_CMI.mat")
            mat_dict = scipy.io.loadmat(patient_id + "CMI_5m_new.mat")

            mat_dict[str(i) + "_" + truncated_filename] = mi_list
            # scipy.io.savemat(patient_id + "corrected_CMI.mat", mat_dict)
            scipy.io.savemat(patient_id + "CMI_5m_new", mat_dict)
            print "\tfeatures for this file is saved!"
        print "features for this patient is saved!"
    return np.array(mi_list)


def get_MIIF_features():
    """
    Generates features for all patients locally.
    """
    home_path = "\Users\Robin\Desktop\EpilepsyVIP\data"
    win_len = 300  # seconds
    win_overlap = 270  # seconds
    f_s = float(1e3)  # Hz
    freqband_names, freq_bands = get_freq_bands()
    freq_bands = freq_bands# get all frequency bands
    patients = ['TA533']

    for patient_id in patients:

        data_path = os.path.join(home_path, patient_id)
        # all_files, data_filenames, file_type, seizure_times = analyze_patient_raw(data_path=data_path, f_s=f_s,
        #                                                                           include_awake=True, include_asleep=True,
        #                                                                           patient_id=patient_id, win_len=win_len,
        #                                                                           win_overlap=win_overlap,
        #                                                                           calc_train_local=True)
        all_files, data_filenames, file_type, seizure_times = analyze_patient_raw_single_file(data_path=data_path, f_s=f_s,
                                                                                  patient_id=patient_id,type='interictal',ind=0)

                                                                                  
        # all_files = sio.loadmat(os.path.join(wavelet_path, "wavelet_filtered_data_TA023"), squeeze_me = True)['filtered'] 
                                                                          


        print "data filenames: ", data_filenames


        print data_filenames

        print "================= extracting features for patient: " + patient_id + " ================="
        for i in range(len(all_files)):
            X = all_files[i]
            filename = data_filenames[i]
            X = notch_filter_data(X, 500)
            print "processing file: ", i
            mi_all_mat = extract_CMI_whole_file(X, filename, patient_id, i, freqbands = freq_bands, save_file=True, chunk_len=win_len, chunk_ovlp=win_overlap)
            print mi_all_mat.shape







get_MIIF_features()


# A list of patients and files of interest
# patient       filename                        filetype      seizure time(seconds)
# TA023         'DA001020_1-1+.edf'             long seizure   (56*60 + 52,57*60 + 52)
# TA510         'CA1353FG_1-1.edf'              long seizure   (26*60 + 28,26*60 + 52)
#               'CA1353FL_1-1.edf'              long seizure   (26*60 + 6,26*60 + 36)
#               'CA1353FN_1-1.edf'              long seizure   (55*60 ,  55*60 + 32)
#               'CA1353FQ_1-1.edf'              long seizure   (27*60 + 8,27*60 + 50)
# TA511         'CA129255_1-1.edf'              long seizure   (20 * 60 + 43, 21 * 60 + 19)
#               'CA12925J_1-1.edf'              long seizure   (53 * 60 + 48, 54 * 60 + 24)
#               'CA12925N_1-1.edf'              long seizure    (10*60 + 28,11*60 + 13)




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

# seizure = 1
# patient_id = 'TA533'
#
# win_len = 300  # seconds
# win_overlap = 270  # seconds
# f_s = float(1e3)  # Hz
# h_path = "/Users/Robin/Desktop/EpilepsyVIP/data"
# d_path = os.path.join(h_path, patient_id)
# filename = 'CA1293ZD_1-1+.edf'
#
#
# dict = scipy.io.loadmat(os.path.join("/Users/Robin/Desktop/EpilepsyVIP/DCEpy/Features/BurnsStudy/MI_features", patient_id + "CMI_3m_30shift.mat"))
# print "keys: ", dict.keys()
#
# data_matrix = dict['2_CA1293ZD_1-1+.edf']
# seizure = False
# if seizure:
#     seizure_time = get_seizure_time(patient_id = patient_id, filename =filename)
# else :
#     seizure_time = None
#
#
#
# # visualize MI for a specific frequency. Refer to get_frequency_bands
# # # "theta", "alpha", "beta", "gamma", "high", "very high"
# band_name = "alpha"
# all_band_names, bands = get_freq_bands()
# band_index = all_band_names.index(band_name)
#
#
#
# if seizure_time!= None:
#     seizure_start_window, seizure_end_window = get_seizure_windows(seizure_time, win_len, win_overlap)
#     print "seizure time: ", seizure_time
#     print "seizure windows: ", seizure_start_window, seizure_end_window
#     print data_matrix.shape
#
# # ==================== dimensionality reduction and plot time series ==================================
# # extract MI for each band
# # "theta", "alpha", "beta", "gamma", "high"
# band_data = data_matrix[:, band_index, :, :]
# #
# # plot reduced features for all bands
# # eigen values, eigen value centrality, katz centrality, pagerank centrality
#
# title = "Katz Centrality for Ictal File (" + band_name + ")"
# reduced = []
# for i in range(0, band_data.shape[0]):
#     print band_data.shape
#     sample = band_data[i, :, :]
#     print "shape of sample:", sample.shape
#     reduced.append(compute_katz(sample))
#     # reduced.append(pagerank_centrality(sample))
#     # reduced.append(eigen(sample))
#     # reduced.append(compute_eigen_centrality(sample))
# reduced = np.array(reduced)   # shape should be samples * 6
# dim1 = reduced
# dim1 = reduced[:, 0]
# dim2 = reduced[:, 1]
# dim3 = reduced[:, 2]
# dim4 = reduced[:, 3]
# dim5 = reduced[:, 4]
# dim6 = reduced[:, 5]
# xs = range(1, 1 + band_data.shape[0])    # num window
# plt.plot(xs, dim1)
# plt.plot(xs, dim2)
# plt.plot(xs, dim3)
# plt.plot(xs, dim4)
# plt.plot(xs, dim5)
# plt.plot(xs, dim6)
# plt.xlabel('Number of Windows')
# plt.title(title)
#
# # plot seizure time
# if seizure_time!= None:
#     plt.axvline(x = seizure_start_window, label = 'seizure start', ls = "dashed")
#     plt.axvline(x = seizure_end_window, label = 'seizure end', ls = "dashed")
# plt.legend(['1st Dimension', '2nd Dimension', '3rd Dimension', '4th Dimension', '5th Dimension','6th Dimension'], loc='upper left')
# plt.show()
#
# # ======== plot MI band features between channel i and channel j =====================
# title = "MI for Ictal File Between Channels 1 and 2 (" + patient_id + ")"
# channel_i = 2
# channel_j = 1
# mi_band = []
# mi_bands = []
# band_data = data_matrix[:, :, channel_i, channel_j]
# for i in range(0, band_data.shape[0]):
#
#     mi_band = []
#     for j in range(0, band_data.shape[1]):
#         sample = band_data[i, j]
#         mi_band.append(sample)
#     mi_bands.append(mi_band)
#
# print(np.array(mi_bands))
# mi_bands = np.array(mi_bands)
#
# dim1 = mi_bands[:, 0]
# dim2 = mi_bands[:, 1]
# dim3 = mi_bands[:, 2]
# dim4 = mi_bands[:, 3]
# dim5 = mi_bands[:, 4]
# dim6 = mi_bands[:, 5]
# xs = range(1, 1 + band_data.shape[0])    # num window
# plt.plot(xs, dim1)
# plt.plot(xs, dim2)
# plt.plot(xs, dim3)
# plt.plot(xs, dim4)
# plt.plot(xs, dim5)
# plt.plot(xs, dim6)
# plt.xlabel('Number of Windows')
# plt.title(title)
#
# # plot seizure time
# if seizure_time!= None:
#     plt.axvline(x = seizure_start_window, label = 'seizure start', ls = "dashed")
#     plt.axvline(x = seizure_end_window, label = 'seizure end', ls = "dashed")
# # plt.legend(['1st Dimension', '2nd Dimension', '3rd Dimension', '4th Dimension', '5th Dimension','6th Dimension'], loc='upper left')
# plt.show()


# ==================== plot graph features for all frequency bands ====================================

# "theta", "alpha", "beta", "gamma", "high"
# p = ax.pcolormesh(data_matrix[23, 0, :, :])       # 340*10/60 = 56 min


#
import matplotlib.ticker as ticker
# fig, axes = plt.subplots(nrows=2, ncols=3)      # 36
# image_idx =30
# #
#
# print data_matrix[image_idx, 0, :, :]
# vmin = 0
# vmax = 1

# "theta"
# p0 = axes[0, 0].pcolormesh(data_matrix[image_idx, 0, :, :], vmin= vmin, vmax=vmax)
# axes[0, 0].set_xticklabels('')
# axes[0, 0].set_xticks([0.5, 1.5,2.5,3.5,4.5,5.5],      minor=True)
# axes[0, 0].set_xticklabels(['1','2','3','4','5', '6'], minor=True)
# # axes[0, 0].set_xlabel('channels')
# # axes[0, 0].set_ylabel('channels')
# axes[0, 0].set_title('Theta')
# print "eigen:  ", eigen(data_matrix[image_idx, 0, :, :])
# print "katz: ",compute_katz(data_matrix[image_idx, 0, :, :])
# print "eigen: ", compute_eigen_centrality(data_matrix[image_idx, 0, :, :])
# # print "subgraph: ", betweenness_centrality(data_matrix[image_idx, 0, :, :])
# print "pagerank: ", pagerank_centrality(data_matrix[image_idx, 0, :, :])
# print "\n"
# fig.colorbar(p0)
#
#
# p1 = axes[0, 1].pcolormesh(data_matrix[image_idx, 1, :, :], vmin=vmin, vmax=vmax)
# axes[0, 1].set_title('Alpha')
# axes[0, 1].set_xticklabels('')
# axes[0, 1].set_xticks([0.5, 1.5,2.5,3.5,4.5,5.5],      minor=True)
# axes[0, 1].set_xticklabels(['1','2','3','4','5', '6'], minor=True)
# # axes[0, 1].set_xlabel('channels')
# # axes[0, 1].set_ylabel('channels')
# print "eigen:  ", eigen(data_matrix[image_idx, 1, :, :])
# print "katz: ",compute_katz(data_matrix[image_idx, 1, :, :])
# print "eigen: ", compute_eigen_centrality(data_matrix[image_idx, 1, :, :])
# print "pagerank: ", pagerank_centrality(data_matrix[image_idx, 1, :, :])
# print "\n"
# # fig.colorbar(p1)
#
#
# p2 = axes[0, 2].pcolormesh(data_matrix[image_idx, 2, :, :], vmin=vmin, vmax=vmax)
# axes[0, 2].set_title('Beta')
# axes[0, 2].set_xticklabels('')
# # axes[0, 2].set_xlabel('channels')
# # axes[0, 2].set_ylabel('channels')
# axes[0, 2].set_xticks([0.5, 1.5,2.5,3.5,4.5,5.5],      minor=True)
# axes[0, 2].set_xticklabels(['1','2','3','4','5', '6'], minor=True)
# print "eigen:  ", eigen(data_matrix[image_idx, 2, :, :])
# print "katz: ",compute_katz(data_matrix[image_idx, 2, :, :])
# print "eigen: ", compute_eigen_centrality(data_matrix[image_idx, 2, :, :])
# print "pagerank: ", pagerank_centrality(data_matrix[image_idx, 2, :, :])
# print "\n"
# # fig.colorbar(p2)
#
#
# p3 = axes[1, 0].pcolormesh(data_matrix[image_idx, 3, :, :], vmin=vmin, vmax=vmax)
# axes[1, 0].set_title('Gamma')
# axes[1, 0].set_xticklabels('')
# # axes[1, 0].set_xlabel('channels')
# # axes[1, 0].set_ylabel('channels')
# axes[1, 0].set_xticks([0.5, 1.5,2.5,3.5,4.5,5.5],      minor=True)
# axes[1, 0].set_xticklabels(['1','2','3','4','5', '6'], minor=True)
# print "eigen:  ", eigen(data_matrix[image_idx, 3, :, :])
# print "katz: ",compute_katz(data_matrix[image_idx, 3, :, :])
# print "eigen: ", compute_eigen_centrality(data_matrix[image_idx, 3, :, :])
# print "pagerank: ", pagerank_centrality(data_matrix[image_idx, 3, :, :])
# print "\n"
# # fig.colorbar(p3)
#
# p4 = axes[1, 1].pcolormesh(data_matrix[image_idx, 4, :, :], vmin=vmin, vmax=vmax)
# axes[1, 1].set_title('High')
# axes[1, 1].set_xticklabels('')
# # axes[1, 1].set_xlabel('channels')
# # axes[1, 1].set_ylabel('channels')
# axes[1, 1].set_xticks([0.5, 1.5,2.5,3.5,4.5,5.5],      minor=True)
# axes[1, 1].set_xticklabels(['1','2','3','4','5', '6'], minor=True)
# print "eigen:  ", eigen(data_matrix[image_idx, 4, :, :])
# print "katz: ",compute_katz(data_matrix[image_idx, 4, :, :])
# print "eigen: ", compute_eigen_centrality(data_matrix[image_idx, 4, :, :])
# print "pagerank: ", pagerank_centrality(data_matrix[image_idx, 4, :, :])
# print "\n"
# # fig.colorbar(p4)
# # #
# # # # plt.subplot(2, 3, 6)
# # # p5 = axes[1, 2].pcolormesh(data_matrix[image_idx, 5, :, :],vmin=vmin, vmax=vmax )
# # # axes[1, 2].set_title('Very High')
# # # print "eigen:  ", eigen(data_matrix[image_idx, 5, :, :])
# # # print "katz: ",compute_katz(data_matrix[image_idx, 5, :, :])
# # # print "eigen: ", compute_eigen_centrality(data_matrix[image_idx, 5, :, :])
# # # print "pagerank: ", pagerank_centrality(data_matrix[image_idx, 5, :, :])
# # # print "\n"
# # # # fig.colorbar(p5)
# plt.show()
# # #
# #
# #
