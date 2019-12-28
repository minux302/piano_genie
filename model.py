import tensorflow.compat.v1 as tf


class PianoGenieModel():
    def __init__(self,
                 config,
                 is_training):
        self.config = config
        self.batch_size = config.batch_size
        self.seq_len = config.seq_len
        self.is_training = is_training

        self.rnn_nunits = config.rnn_nunits
        self.rnn_nlayers = config.rnn_nlayers

    def placeholders(self):
        with tf.variable_scope("input"):
            pitches = tf.placeholder(tf.int32,
                                     [None, self.seq_len],
                                     name="pitches")
            delta_times_int = tf.placeholder(tf.int32,
                                             [None, self.seq_len],
                                             name="delta_times_int")
        return {"pitches": pitches,
                "delta_times_int": delta_times_int}

    def _lstm_encoder(self, x):
        with tf.variable_scope("lstm_encoder"):
            x = tf.layers.dense(x, self.rnn_nunits)
            x = tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(self.rnn_nunits,
                                         return_sequences=True))(x)
            x = tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(self.rnn_nunits,
                                         return_sequences=True))(x)  # (batch_size, seq_len, self.rnn_nunits * 2)
            x = tf.layers.dense(x, 1)
        return x

    def _lstm_decoder(self, x):
        with tf.variable_scope("lstm_decoder"):
            x = tf.layers.dense(x, self.rnn_nunits)
            x = tf.keras.layers.LSTM(self.rnn_nunits,
                                     return_sequences=True)(x)
            x = tf.keras.layers.LSTM(self.rnn_nunits,
                                     return_sequences=True)(x)
            x = tf.layers.dense(x, 88)
        return x

    def _iqae(self, x):
        eps = 1e-7
        scale = float(self.config.iqae_nbins - 1)  # Todo: why use scale ? maybe not need.
        with tf.variable_scope("iqae"):
            hard_sigmoid_x = tf.clip_by_value((x + 1) / 2.0, -eps, 1 + eps)
            _quantized_x = tf.round(scale * hard_sigmoid_x)
            quantized_x = 2 * (_quantized_x / scale) - 1
        return x + tf.stop_gradient(quantized_x - x)

    def _range_loss(self, enc_outputs):
        return tf.reduce_mean(
            tf.square(tf.maximum(tf.abs(enc_outputs) - 1, 0)))

    def _contour_loss(self, enc_outputs, input_pitches):
        enc_outputs = tf.squeeze(enc_outputs, axis=2)  # (batch_size, seq_len)
        delta_enc_outputs = enc_outputs[:, 1:] - enc_outputs[:, :-1]
        delta_pitches = tf.cast(input_pitches[:, 1:] - input_pitches[:, :-1],
                                tf.float32)
        return tf.reduce_mean(tf.square(
            tf.maximum(1.0 - tf.multiply(delta_enc_outputs, delta_pitches),
                       0)))

    def _reconstruction_loss(self, dec_outputs, input_pitches):
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=dec_outputs,
                labels=input_pitches))

    def build(self, input_dict):
        input_pitches = input_dict["pitches"]  # (batch_size, seq_len)
        input_delta_times_int = input_dict["delta_times_int"]

        enc_inputs = tf.concat(
            [tf.one_hot(input_pitches, 88),
             tf.one_hot(input_delta_times_int,
                        self.config.max_discrete_times + 1)],
            axis=2)  # (batch_size, seq_len, 88 + max_discrete_times + 1)

        enc_outputs = self._lstm_encoder(enc_inputs)  # (batch_size, seq_len, 1)
        quantized_enc_outputs = self._iqae(enc_outputs)  # (batch_size, seq_len, 1)
        dec_outputs = self._lstm_decoder(quantized_enc_outputs)  # (batch_size, seq_len, 88)

        range_loss = self._range_loss(enc_outputs)
        contour_loss = self._contour_loss(enc_outputs, input_pitches)
        reconstruction_loss = self._reconstruction_loss(dec_outputs,
                                                        input_pitches)

        return {"dec_outputs": dec_outputs,
                "range_loss": range_loss,
                "contour_loss": contour_loss,
                "reconstructions_loss": reconstruction_loss}

    def loss(self, output_dict):
        combined_loss = output_dict["reconstruction_loss"] + \
            self.config.range_loss_ratio * output_dict["range_loss"] + \
            self.config.coutour_loss_ratio * output_dict["contour_loss"]
        return combined_loss
