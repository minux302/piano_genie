import util
import tensorflow.compat.v1 as tf
from tensorflow.contrib import rnn as contrib_rnn


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
            delta_times = tf.placeholder(tf.float32,
                                         [None, self.seq_len],
                                         name="delta_times")
            start_times = tf.placeholder(tf.float32,
                                         [None, self.seq_len],
                                         name="start_times")
            end_times = tf.placeholder(tf.float32,
                                       [None, self.seq_len],
                                       name="end_times")
        return {"pitches": pitches,
                "delta_times": delta_times,
                "start_times": start_times,
                "end_times": end_times}

    def _lstm_encoder(self, x):
        with tf.variable_scope("lstm_encoder"):
            x = tf.layers.dense(x, self.rnn_nunits)
            x = tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(self.rnn_nunits,
                                         return_sequences=True))(x)
            x = tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(self.rnn_nunits,
                                         return_sequences=True))(x)
        return x

    def _iqae_quantizer(self, x):
        eps = 1e-7
        scale = float(self.config.iqae_nbins - 1)  # Todo, why use scale ?
        with tf.variable_scope("iqae_quantizer"):
            hard_sigmoid_x = tf.clip_by_value((x + 1) / 2.0, -eps, 1 + eps)
            _quantized_x = tf.round(scale * hard_sigmoid_x)
            quantized_x = 2 * (_quantized_x / scale) - 1
            forward_quantized_x = x + tf.stop_gradient(quantized_x - x)
        return tf.expand_dims(forward_quantized_x, axis=2)

    def build(self, input_dict):
        # output_dict = {}
        input_pitches = input_dict["pitches"]
        # input_delta_times = input_dict["delta_times"]
        # input_start_times = input_dict["start_times"]
        # input_end_times = input_dict["end_times"]

        pitches = util.demidity(input_pitches)
        # pitches_scalar = ((tf.cast(pitches, tf.float32) / 87.) * 2.) - 1.

        # seq_lens = tf.ones([self.batch_size], dtype=tf.int32) * self.seq_len  # (1,)

        enc_inputs = tf.one_hot(pitches, 88, axis=-1)  # (batch_size, seq_len, 88)
        enc_outputs = self._lstm_encoder(enc_inputs)  # (batch_size, seq_len, rnn_nunits * 2)

        pre_iqae_enc = tf.layers.dense(enc_outputs, 1)[:, :, 0]  # (batch_size, seq_len)
        output = self._iqae_quantizer(pre_iqae_enc)  # (batch_size, seq_len, 1)
        return output
        """
                # Regularize to encourage encoder to output in range
                stp_emb_iq_range_penalty = tf.reduce_mean(
                    tf.square(tf.maximum(tf.abs(pre_iq_encoding) - 1, 0)))

                # Regularize to encourage encoder to output in range
                stp_emb_iq_dlatents = pre_iq_encoding[:, 1:] - pre_iq_encoding[:, :-1]
                stp_emb_iq_dnotes = tf.cast(pitches_scalar[:, 1:] - pitches_scalar[:, :-1],
                                            tf.float32)

                stp_emb_iq_contour_penalty = tf.reduce_mean(
                    tf.square(
                        tf.maximum(
                            1.0 - tf.multiply(
                                stp_emb_iq_dnotes, stp_emb_iq_dlatents), 0)))

                # Determine which elements round to valid indices
                stp_emb_iq_inrange = tf.logical_and(
                    tf.greater_equal(pre_iq_encoding, -1),
                    tf.less_equal(pre_iq_encoding, 1))
                stp_emb_iq_inrange_mask = tf.cast(stp_emb_iq_inrange, tf.float32)

            out_dict["stp_emb_iq_quantized"] = stp_emb_iq_quantized  # need
            out_dict["stp_emb_iq_discrete"] = stp_emb_iq_discrete
            out_dict["stp_emb_iq_range_penalty"] = stp_emb_iq_range_penalty  # need
            out_dict["stp_emb_iq_contour_penalty"] = stp_emb_iq_contour_penalty  # need
            latents.append(stp_emb_iq_quantized)

        """
        """
        # Decode
        with tf.variable_scope("decoder"):
            dec_stp, dec_initial_state, dec_final_state = simple_lstm_decoder(
                dec_feats,
                seq_lens,
                batch_size,
                rnn_nlayers=cfg.rnn_nlayers,
                rnn_nunits=cfg.rnn_nunits)

            with tf.variable_scope("pitches"):
                dec_recons_logits = tf.layers.dense(dec_stp, 88)

            dec_recons_loss = weighted_avg(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=dec_recons_logits,
                    labels=pitches),
                stq_varlen_mask)

            out_dict["dec_initial_state"] = dec_initial_state
            out_dict["dec_final_state"] = dec_final_state
            out_dict["dec_recons_logits"] = dec_recons_logits
            out_dict["dec_recons_scores"] = tf.nn.softmax(dec_recons_logits, axis=-1)
            out_dict["dec_recons_preds"] = tf.argmax(
                dec_recons_logits, output_type=tf.int32, axis=-1)
            out_dict["dec_recons_midi_preds"] = util.remidify(
                out_dict["dec_recons_preds"])
            out_dict["dec_recons_loss"] = dec_recons_loss

        discrete = out_dict["stp_emb_iq_discrete"]
        dx = pitches[:, 1:] - pitches[:, :-1]
        dy = discrete[:, 1:] - discrete[:, :-1]
        contour_violation = tf.reduce_mean(tf.cast(tf.less(dx * dy, 0), tf.float32))

        dx_hold = tf.equal(dx, 0)
        deviate_violation = weighted_avg(
            tf.cast(tf.not_equal(dy, 0), tf.float32), tf.cast(dx_hold, tf.float32))

        out_dict["contour_violation"] = contour_violation
        out_dict["deviate_violation"] = deviate_violation

        return out_dict
        """
