class Config():
    def __init__(self):
        self.file_name = 'midi_sample_tf'
        self.max_discrete_times = 32

        # training
        self.batch_size = 2
        self.seq_len = 5
        self.range_loss_ratio = 1.
        self.contour_loss_ratio = 1.

        # model
        self.rnn_nlayers = 2
        self.rnn_nunits = 128
        self.iqae_nbins = 8
