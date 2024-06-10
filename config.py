class Config(object):
    
    def __init__(self):
        self.sub_seq_len = 10  # subsequence length
        self.nsubseq = 8 #10

        self.learning_rate = 1e-4
        self.training_epoch = 10 #10*self.sub_seq_len*self.nsubseq
        self.batch_size = 32 #8

        # spectrogram size
        self.ndim = 129  # freq bins
        self.frame_seq_len = 17 #29  # time frames in one sleep epoch spectrogram

        #self.artifacts_label= # check and change data
        self.nclasses_model=1
        self.nclasses_data=4 #2

        self.mask_artifacts=True #"whether masking artifacts in loss")
        self.artifact_detection=False

        self.latent_dim=100