def preprocessing(self, input):
        # input of shape [-1, ndim, nchannel]
        # triangular filterbank shape
        Wbl = tf.constant(self.filtershape.lin_tri_filter_shape(nfilt=self.config.nfilter,
                                                                nfft=self.config.nfft,
                                                                samplerate=self.config.samplerate,
                                                                lowfreq=self.config.lowfreq,
                                                                highfreq=self.config.highfreq),
                                                                dtype=tf.float32,
                                                                name="W-filter-shape-eeg")

        # filter bank layer for eeg
        # Temporarily crush the feature_mat's dimensions
        Xeeg = tf.reshape(tf.squeeze(input[:, :, :, 0]), [-1, self.config.ndim])
        # first filter bank layer
        Weeg = tf.Variable(initial_value=tf.random.normal(shape=[self.config.ndim, self.config.nfilter]),name="Weeg")

        # non-negative constraints
        Weeg = tf.sigmoid(Weeg)
        Wfb_eeg = tf.multiply(Weeg, Wbl)
        HWeeg = tf.matmul(Xeeg, Wfb_eeg)  # filtering

        # filter bank layer for eog
        Xeog = tf.reshape(tf.squeeze(input[:, :, :, 1]), [-1, self.config.ndim])
        # first filter bank layer
        Weog = tf.Variable(initial_value=tf.random.normal(shape=[self.config.ndim, self.config.nfilter]),name="Weog")
        # non-negative constraints
        Weog = tf.sigmoid(Weog)
        Wfb_eog = tf.multiply(Weog, Wbl)
        HWeog = tf.matmul(Xeog, Wfb_eog)  # filtering

        # filter bank layer for emg
        Xemg = tf.reshape(tf.squeeze(input[:, :, :, 2]), [-1, self.config.ndim])
        # first filter bank layer
        Wemg = tf.Variable(initial_value=tf.random.normal(shape=[self.config.ndim, self.config.nfilter]),name="Wemg")
        # non-negative constraints
        Wemg = tf.sigmoid(Wemg)
        Wfb_emg = tf.multiply(Wemg, Wbl)
        HWemg = tf.matmul(Xemg, Wfb_emg)  # filtering

        X2 = tf.concat([HWeeg, HWeog, HWemg], axis=1)

        return X2