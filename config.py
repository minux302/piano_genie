class Config():
    def __init__(self):
        self.file_name = 'midi_sample_tf'

        self.batch_size = 2
        self.seq_len = 5

        self.rnn_nlayers = 2
        self.rnn_nunits = 128

        self.stp_emb_iq_nbins = 8
