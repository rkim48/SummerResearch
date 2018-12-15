import os, sys, time
import time
from scipy import special
from DCEpy.Features.BurnsStudy.Permutation_MI.fast_analyze_patient_raw import fast_analyze_patient_raw
import winsound
import json
from preprocessing import notch_filter_data
from multiprocessing import Pool
import scipy.io
from multiprocessing import cpu_count



# to_prepr = "/home/rk48/Desktop/EpilepsyVIP/DCEpy/Features/Preprocessing/"
# to_edf = "/home/rk48/Desktop/EpilepsyVIP/DCEpy/General/DataInterfacing/"
# to_burns = "/home/rk48/Desktop/EpilepsyVIP/DCEpy/Features/BurnsStudy"
# sys.path.append(to_prepr)
# sys.path.append(to_edf)
# sys.path.append(to_burns)

# from MI_pipeline import *
# os.chdir(to_edf)
# cwd = os.getcwd()
# os.chdir(to_burns)
from DCEpy.Features.BurnsStudy.ictal_inhibitors_final import analyze_patient_raw
import numpy.random as nr
import scipy.spatial as spatial
import scipy.signal
import numpy as np
import dill as pickle
from pathos.multiprocessing import ProcessPool as Pool
import pathos.multiprocessing as mp

# os.chdir(cwd)
import scipy
import math

# Number 1: create json files from raw data
# Number 2: try to multiprocess different parts of feature generation pipeline

def get_freq_bands_new():
    return ["delta", "theta", "alpha", "beta"], [[0.5,4], [4,8], [8,13], [13, 35]]

def mutual_information_ksg2004_XY(x, y, k=3, seed=1, threshold=True):
    """" Mutual information between x and y in nats. Implementation by Rakesh. """
    # tol - small noise to break degeneracy, see doc.
    tol = 1e-10
    assert len(x) == len(y)
    assert k <= len(x) - 1  # to ensure k^th nearest neighbor exists for each point in the x and y

    nr.seed(seed=seed)
    x = [list(p + tol * nr.rand(len(x[0]))) for p in x]
    y = [list(p + tol * nr.rand(len(y[0]))) for p in y]

    data = [list(x[ii] + y[ii]) for ii in range(len(x))]

    # Build a kd-tree data structure for the data in joint space and in marginal subspaces, to
    # ensure faster k-nn queries
    xy_tree = spatial.cKDTree(data)
    x_tree = spatial.cKDTree(x)
    y_tree = spatial.cKDTree(y)
    # Compute the distance between each data point and its kth nearest neighbor
    xy_dist = [xy_tree.query(elem, k=k + 1, p=float('inf'))[0][k] for elem in data]
    x_num = [len(x_tree.query_ball_point(x[ii], xy_dist[ii] - 1e-15, p=float('inf'))) for ii in
             range(len(x))]
    y_num = [len(y_tree.query_ball_point(y[ii], xy_dist[ii] - 1e-15, p=float('inf'))) for ii in
             range(len(y))]

    mi_est = special.digamma(k) + special.digamma(len(x)) - (sum(special.digamma(x_num))
                                                             + sum(special.digamma(y_num))) / len(x)

    if threshold:
        if mi_est < 0:
            mi_est = 0

    return mi_est


def mi_in_frequency_channel(dXband, dYband):
    ndf = dXband.shape[1]  # ndf is length of dXband, specific frequency band of the DFT
    mi_band = []
    for i in range(ndf):

        real_dXi = np.hstack((np.real(dXband[:, [i]]), np.imag(dXband[:, [i]])))
        real_dYi = np.hstack((np.real(dYband[:, [i]]), np.imag(dYband[:, [i]])))
        mi_band.append(mutual_information_ksg2004_XY(real_dXi, real_dYi, k=3))


    return np.array(mi_band)


def cross_channel_mutual_information_in_frequency(X, Y, mi_window_length, normalization, f_s=1000):
    """
    Computes mutual information between two signals for the selected frequency bands.
    X: NxNs matrix, where N is the number of windows per chunk and Ns is the length of a window.
       each signal can be from a single channel or the average of all channels.
    Y:  NxNs matrix, where N is the number of windows per chunk and Ns is the length of a window.
       each signal can be from a single channel or the average of all channels.
    mi_window_length: determines maximum possible length of FFT excluding zero-padding; this determines our FFT resolution
    normalization: determines scaling of FFT, default is 'None' (scales by 1/N) and other option is 'ortho' (scales by 1/sqrt(N))
    f_s: sampling frequency/rate is 1000 Hz
    """

    # Resolution variable not needed but tells one how fine the FFT bins are
    # Since these bins are averaged over a specific frequency band in process of feature generation, FFT resolution will play a role
    # Prefer not setting length of FFT over length of window which will automatically zero-pad. Zero-padding provides
    # "fake" resolution and only interpolates between actual FFT bins

    resolution = int(f_s / mi_window_length)
    dX = np.fft.fft(X, n=mi_window_length, axis=1, norm=normalization)
    dY = np.fft.fft(Y, n=mi_window_length, axis=1, norm=normalization)
    frequencies = np.fft.fftfreq(mi_window_length) * f_s

    list_mi_bands = []
    _, freqbands = get_freq_bands_new()
    for freqband in freqbands:
        # select the FFT within a frequency band
        freq_indices = np.where((freqband[0] <= frequencies) & (frequencies <= freqband[1]))[0]
        dXband = dX[:, freq_indices]
        dYband = dY[:, freq_indices]

        # compute cross channel MI in frequency
        mi_band = mi_in_frequency_channel(dXband, dYband)

        list_mi_bands.append(np.mean(mi_band))

    # print "MI of the band: ", list_mi_bands
    return list_mi_bands


def cross_channel_MI(X_chunk, mi_win_len, normalization):
    """
    Returns cross channel mutual information for all frequency bands. Should be
    (number of channels, number of channels, number of frequency bands)
    """
    # for every two channels
    N, Ns, p = X_chunk.shape
    _, freqbands = get_freq_bands_new()
    cmi_graph = np.zeros(shape=(p, p, len(freqbands)))

    for i in xrange(p):
        for j in xrange(i, p):
            X_channel_i = X_chunk[:, :, i]
            X_channel_j = X_chunk[:, :, j]
            if i == j:
                cmi_graph[i, j, :] = np.zeros(shape=(len(freqbands),))
            else:
                cmi_graph[i, j, :] = cross_channel_mutual_information_in_frequency(X_channel_i, X_channel_j, mi_win_len, normalization)

                cmi_graph[j, i, :] = cmi_graph[i, j, :]


    list_mis_allbands = []
    for i in range(len(freqbands)):
        list_mis_allbands.append(cmi_graph[:, :, i])

    # shape should be number of (frequencies, number of channels, number of channels)
    return np.array(list_mis_allbands)


def window_data(data, win_len, win_ovlap, f_s=1000):
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

        # get window of data and apply Hanning window if overlap is not zero
        if end <= n:
            if win_ovlap == 0:
                window_of_data = data[int(start):int(end), :]
            else:
                window_of_data = data[int(start):int(end), :] * np.hanning(int(end) - int(start))[:, None]

            if window_of_data.shape[0] < win_len:  # get rid of short windows in the end
                break
            all_windowed_data.append(window_of_data)
    return np.array(all_windowed_data)


def extract_CMI_whole_file(X, matfile_name, filename, patient_id, i, chunk_len,
                           chunk_ovlp, mi_win_len, mi_win_overlap, normalization, save_file=False):

    mi_list = []

    # window data
    chunked_data = window_data(X, win_len=chunk_len, win_ovlap=chunk_ovlp)

    # compute MI information for each chunk
    for chunk_idx in range(0, chunked_data.shape[0]):
        chunk = chunked_data[chunk_idx, :, :]
        windowed_chunk = window_data(chunk, win_len=mi_win_len, win_ovlap=mi_win_overlap)
        start_chunk = time.time()
        mi_all_freqs = cross_channel_MI(windowed_chunk, mi_win_len, normalization)
        end_chunk = time.time()
        chunk_time = end_chunk - start_chunk
        print "Time to compute feature for one window: ", chunk_time
        print "Estimated time to compute features for this file: ", chunk_time * chunked_data.shape[0]
        if chunk_idx == 0:
            print '\a'
        mi_list.append(mi_all_freqs)

    print "\t MI matrix shape (should be 6*6*6): ", mi_all_freqs.shape

    if save_file == True:
        index = filename.rfind('/')
        truncated_filename = filename[index + 1:]
        if i == 1:
            scipy.io.savemat(patient_id + matfile_name, {str(i) + "_" + truncated_filename: mi_list})

        else:
            mat_dict = scipy.io.loadmat(patient_id + matfile_name)
            mat_dict[str(i) + "_" + truncated_filename] = mi_list
            scipy.io.savemat(patient_id + matfile_name, mat_dict)
            print "\tfeatures for file " + str(i) + " is saved!"
        print "features for this patient is saved!"
    return np.array(mi_list)


def get_MIIF_features():
    """
    Generates features for all patients locally.
    """
    win_len = 120  # seconds
    win_overlap = 90  # seconds
    mi_len = 2
    mi_ovlap = 1
    f_s = float(1e3)  # Hz
    patient_id = 'TA533'
    matfile_name = "MULTI_TEST"
    stormtrooper = False

    if stormtrooper == True:
        home_path = '/media/ExtHDD2/rk48Ext/data'
    else:
        home_path = "\Users\Robin\Desktop\EpilepsyVIP\data"

    data_path = os.path.join(home_path, patient_id)

    # open JSON serialized raw patient data and load into all_files list
    # from 6 minutes to 38 seconds (9x speedup)
    with open("TA533_files.json", "r") as read_file:
        data = json.load(read_file)
    all_files = []
    for i in xrange(len(data)):
        all_files.append(np.asarray(data[i]))
    # grab the names of files, file types, and seizure times for ictal files
    _, data_filenames, file_type, seizure_times = fast_analyze_patient_raw(data_path)

    print "================= extracting features for patient: " + patient_id + " ================="

    # from multiprocessing.pool import ThreadPool as Pool
    # from multiprocessing.pool import Pool
    return all_files, matfile_name, data_filenames, patient_id, win_len, win_overlap, mi_len, mi_ovlap


    # for i in range(len(all_files)):
    #     X = all_files[i]
    #     filename = data_filenames[i]
    #     X = notch_filter_data(X, 500)
    #     print "processing file: ", i
    #     mi_all_mat = extract_CMI_whole_file(X, matfile_name=matfile_name, filename=filename, patient_id=patient_id,
    #                                         i=i, chunk_len=win_len,
    #                                         chunk_ovlp=win_overlap, mi_win_len=mi_len, mi_win_overlap=mi_ovlap,
    #                                         normalization='ortho', save_file=True )
    #     print mi_all_mat.shape

def wrapper(args):
    return extract_CMI_whole_file(*args)

def multi():
    from multiprocessing import Pool
    # begin = time.time()
    all_files, matfile_name, data_filenames, patient_id, win_len, win_overlap, mi_len, mi_ovlap = get_MIIF_features()
    args = [(notch_filter_data(all_files[i], 500), matfile_name, data_filenames[i], patient_id, i, win_len, win_overlap,
             mi_len, mi_ovlap, 'ortho', True) for i in range(len(all_files))]
    print args[1]
    pool = Pool(4)
    pool.map(wrapper, args)
    pool.close()
    pool.join()


    # print time.time() - begin

    duration = 500  # millisecond
    freq = 440  # Hz
    winsound.Beep(freq, duration)

if __name__ == '__main__':
    multi()




