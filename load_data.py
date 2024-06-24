import numpy as np
import h5py
from dataGen import DataGenerator3

class load_data:
    def __init__(self, eeg_filelist=None, eog_filelist=None, emg_filelist=None, data_shape_2=np.array([29, 128]), seq_len = 20, nclasses = 4, artifact_detection = False, artifacts_label = 4, shuffle=False):

        # Init params

        self.eeg_list_of_files = []
        self.eog_list_of_files = []
        self.emg_list_of_files = []
        self.file_sizes = []

        self.data_shape_2 = data_shape_2
        self.data_shape_1 = np.array([512])

        self.seq_len = seq_len
        self.Ncat = nclasses
        self.artifact_detection = artifact_detection
        self.artifacts_label = artifacts_label

        self.shuffle = shuffle

        # list of files and their size
        if eeg_filelist is not None:
            self.eeg_list_of_files, self.file_sizes = self.read_file_list(eeg_filelist)
        if eog_filelist is not None:
            self.eog_list_of_files, _ = self.read_file_list(eog_filelist)
        if emg_filelist is not None:
            self.emg_list_of_files, _ = self.read_file_list(emg_filelist)

        self.eeg_meanX1, self.eeg_stdX1 = self.load_data_compute_norm_params_by_signal_X1(self.eeg_list_of_files)
        self.eog_meanX1, self.eog_stdX1 = self.load_data_compute_norm_params_by_signal_X1(self.eog_list_of_files)
        self.emg_meanX1, self.emg_stdX1 = self.load_data_compute_norm_params_by_signal_X1(self.emg_list_of_files)

        self.eeg_meanX2, self.eeg_stdX2 = self.load_data_compute_norm_params_by_signal_X2(self.eeg_list_of_files)
        self.eog_meanX2, self.eog_stdX2 = self.load_data_compute_norm_params_by_signal_X2(self.eog_list_of_files)
        self.emg_meanX2, self.emg_stdX2 = self.load_data_compute_norm_params_by_signal_X2(self.emg_list_of_files)

        self.eeg_data= DataGenerator3(self.eeg_list_of_files,
                                 self.file_sizes,
                                 data_shape_2=self.data_shape_2,
                                 seq_len=self.seq_len,
                                 Ncat=self.Ncat, 
                                 artifact_detection=self.artifact_detection,
                                 artifacts_label=self.artifacts_label)
        
        self.eeg_data.normalize_by_signal_X1(self.eeg_meanX1, self.eeg_stdX1)
        self.eeg_data.normalize_by_signal_X2(self.eeg_meanX2, self.eeg_stdX2)

        self.eog_data= DataGenerator3(self.eog_list_of_files,
                                 self.file_sizes,
                                 data_shape_2=self.data_shape_2,
                                 seq_len=self.seq_len,
                                 Ncat=self.Ncat, 
                                 artifact_detection=self.artifact_detection, 
                                 artifacts_label=self.artifacts_label)
        
        self.eog_data.normalize_by_signal_X1(self.eog_meanX1, self.eog_stdX1)
        self.eog_data.normalize_by_signal_X2(self.eog_meanX2, self.eog_stdX2)

        self.emg_data= DataGenerator3(self.emg_list_of_files,
                                 self.file_sizes,
                                 data_shape_2=self.data_shape_2,
                                 seq_len=self.seq_len,
                                 Ncat=self.Ncat, 
                                 artifact_detection=self.artifact_detection, 
                                 artifacts_label=self.artifacts_label)
        
        self.emg_data.normalize_by_signal_X1(self.emg_meanX1, self.emg_stdX1)
        self.emg_data.normalize_by_signal_X2(self.emg_meanX2, self.emg_stdX2)

        self.y = np.stack((self.eeg_data.y, self.eog_data.y, self.emg_data.y), axis=-1)
        self.X1 = np.stack((self.eeg_data.X1, self.eog_data.X1, self.emg_data.X1), axis=-1)
        self.X2 = np.stack((self.eeg_data.X2, self.eog_data.X2, self.emg_data.X2), axis=-1) # merge and make new dimension
        self.label = np.stack((self.eeg_data.label,self.eog_data.label,self.emg_data.label), axis=-1)

    # read in a list of file
    def read_file_list(self, filelist):
        list_of_files = []
        file_sizes = []
        with open(filelist) as f:
            lines = f.readlines()
            for l in lines:
                print(l)
                items = l.split()
                list_of_files.append(items[0])
                file_sizes.append(int(items[1]))
        return list_of_files, file_sizes

    # read data from mat files in the list stored in the file 'filelist'
    # and compute normalization parameters on the flight
    def load_data_compute_norm_params_by_signal_X1(self, list_of_files):
        meanX = None
        meanXsquared = None
        count = 0
        print('Computing normalization parameters')
        means = {}
        stds = {}
        for i in range(len(list_of_files)):
            X1 = self.read_X1_from_mat_file(list_of_files[i].strip())
            Ni = len(X1)
            meanX_i = X1.mean(axis=0)
            stdX_i = X1.std(axis=0)
            means[list_of_files[i]] = meanX_i
            stds[list_of_files[i]] = stdX_i

        return means, stds
    
    def load_data_compute_norm_params_by_signal_X2(self, list_of_files):
        meanX = None
        meanXsquared = None
        count = 0
        print('Computing normalization parameters')
        means = {}
        stds = {}
        for i in range(len(list_of_files)):
            X2 = self.read_X2_from_mat_file(list_of_files[i].strip())
            Ni = len(X2)
            X2 = np.reshape(X2,(Ni*self.data_shape_2[0], self.data_shape_2[1]))
            meanX_i = X2.mean(axis=0)
            stdX_i = X2.std(axis=0)
            means[list_of_files[i]] = meanX_i
            stds[list_of_files[i]] = stdX_i

        return means, stds

    def read_X1_from_mat_file(self,filename):
        """
        Read in X1 data from a data file in mat file HD5F file
        """
        # Load data
        print(filename)
        data = h5py.File(filename,'r')
        data.keys()
        X1 = np.array(data['X1']) # time-frequency input
        X1 = np.transpose(X1, (1, 0))  # rearrange dimension
        return X1
    
    def read_X2_from_mat_file(self,filename):
        """
        Read in X2 data from a data file in mat file HD5F file
        """
        # Load data
        print(filename)
        data = h5py.File(filename,'r')
        data.keys()
        X2 = np.array(data['X2']) # time-frequency input
        X2 = np.transpose(X2, (2, 1, 0))  # rearrange dimension
        return X2


