import util
import tensorflow.compat.v1 as tf
from tensorflow.contrib import rnn as contrib_rnn


def simple_lstm_decoder(features,
                        seq_lens,
                        batch_size,
                        rnn_nlayers=2,
                        rnn_nunits=128):
    x = features

    with tf.variable_scope("rnn_input"):
        x = tf.layers.dense(x, rnn_nunits)

    celltype = contrib_rnn.LSTMBlockCell
    cell = contrib_rnn.MultiRNNCell(
        [celltype(rnn_nunits) for _ in range(rnn_nlayers)])
    with tf.variable_scope("rnn"):
        # Todo why here is need ?
        initial_state = cell.zero_state(batch_size, dtype)
        x, final_state = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=x,
            sequence_length=seq_lens,
            initial_state=initial_state)

    return x, initial_state, final_state


def weighted_avg(t, mask=None, axis=None, expand_mask=False):
    if mask is None:
        return tf.reduce_mean(t, axis=axis)
    else:
        if expand_mask:
            maks = tf.expand_dims(mask, axis=-1)
        return tf.reduce_sum(
            tf.multiply(t, mask), axis=axis) / tf.resuce_sum(mask, axis=axis)


class PianoGenirModel():
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
        note_pitches = tf.placeholder(tf.int32,
                                      [None, self.seq_len],
                                      name="note_pitches")
        note_delta_times = tf.placeholder(tf.float32,
                                          [None, self.seq_len],
                                          name="note_delta_times")
        note_start_times = tf.placeholder(tf.float32,
                                          [None, self.seq_len],
                                          name="note_start_times")
        note_end_times = tf.placeholder(tf.float32,
                                        [None, self.seq_len],
                                        name="note_end_times")
        return (note_pitches,
                note_delta_times,
                note_start_times,
                note_end_times)

    # Todo, rethink for this module
    def _lstm_encoder(self, inputs):
        x = tf.layers.dense(inputs, self.rnn_nunits)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.rnn_nunits,
                                                               return_sequences=True))(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.rnn_nunits,
                                                               return_sequences=True))(x)
        return x

    def build(self, inputs):
        out_dict = {}
        note_pitches = inputs[0]
        note_delta_times = inputs[1]
        note_start_times = inputs[2]
        note_end_times = inputs[3]

        # Paese features
        pitches = util.demidity(note_pitches)  # [0, 88)

        # Create sequence lens
        seq_lens = tf.ones([self.batch_size], dtype=tf.int32) * self.seq_len  # shape (1)

        enc_feats = tf.one_hot(pitches, 88, axis=-1)  # (batch_size, seq_len, 88)
        with tf.variable_scope("encoder"):
            enc_stp = self._lstm_encoder(enc_feats)  # (batch_size, seq_len, rnn_nunits * 2)

        # Integer-quantized step embeddings with straight-through
        with tf.variable_scope("stp_emb_iq"):
            # Todo what is pre_iq_encoding ?
            with tf.variable_scope("pre_iq"):
                pre_iq_encoding = tf.layers.dense(enc_stp, 1)[:, :, 0]  # (batch_size, seq_len)

            def iqst(x, n):
                eps = 1e-7
                s = float(n - 1)
                xp = tf.clip_by_value((x + 1) / 2.0, -eps, 1 + eps)
                xpp = tf.round(s * xp)
                xppp = 2 * (xpp / s) - 1
                return xpp, x + tf.stop_gradient(xppp - x)

            with tf.variable_scope("quantizer"):
                # Pass rounded vals to decoder w/ straight-through estimator
                stp_emb_iq_discrete_f, stp_emb_iq_discrete_rescaled = iqst(
                    pre_iq_encoding, cfg.stp_emb_iq_nbins)
                stp_emb_iq_discrete = tf.cast(stq_emb_iq_discrete_f + 1e-4, tf.int32)
                stp_emb_iq_discrete_f = tf.cast(stp_emb_iq_discrete, tf.float32)
                stp_emb_iq_quantized = tf.expand_dims(
                    stp_emb_iq_discrete_rescaled, axis=2)

                # Determine which elements round to valid indices
                stp_emb_iq_inrange = tf.logical_and(
                    tf.greater_equal(pre_iq_encoding, -1),
                    tf.less_equal(pre_iq_encoding, 1))
                stp_emb_iq_inrange_mask = tf.cast(stp_emb_iq_inrange, tf.float32)
                stp_emb_iq_valid_p = weighted_avg(stp_emb_iq_inrange_mask,
                                                  stp_varlen_mask)

                # Regularize to encourage encoder to output in range
                stp_emb_iq_dlatents = pre_iq_encoding[:, 1:] - pre_iq_encoding[:, :-1]
                stp_emb_iq_dnotes = tf.cast(pitches_scalar[:, 1:] - pitches_scalar[:, :-1],
                                            tf.float32)

                power_func = tf.square
                comp_func = tf.multiply

                stp_emb_iq_contour_penalty = weighted_avg(
                    power_func(
                        tf.maximum(
                            cfg.stp_emb_iq_contour_margin - comp_func(
                                stp_emb_iq_dnotes, stp_emb_iq_dlatents), 0)),
                    None if stp_varlen_mask is None else stp_varlen_mask[:, 1:])

                # Regularize to maintain note consistency
                stp_emb_iq_note_held = tf.cast(
                    tf.equal(pitches[:, 1:] - pitches[:, :-1], 0), tf.float32)

                power_func = tf.square
                if stp_varlen_mask is None:
                    mask = stp_emb_iq_note_held
                else:
                    mask = stp_varlen_mask[:, 1:] * stp_emb_iq_note_held
                stp_emb_iq_deviate_penalty = weighted_avg(
                        power_func(stp_emb_iq_dlatents), mask)

                # Calculate perplexity of discrete encoder posterior
                if stp_varlen_mask is None:
                    mask = stp_emb_iq_inrange_mask
                else:
                    mask = stp_varlen_mask * stp_emb_iq_inrange_mask
                stp_emb_iq_avg_probs = weighted_avg(
                    stp_emb_iq_discrete_oh,
                    mask,
                    axis=[0, 1],
                    expand_mask=True)
                stp_emb_iq_discrete_ppl = tf.exp(-tf.reduce_sum(
                    stp_emb_iq_avg_probs * tf.log(stp_emb_iq_avg_probs + 1e-10))

            out_dict["stp_emb_iq_quantized"] = stp_emb_iq_quantized
            out_dict["stp_emb_iq_discrete"] = stp_emb_iq_discrete
            out_dict["stp_emb_iq_valid_p"] = stp_emb_iq_valid_p
            out_dict["stp_emb_iq_range_penalty"] = stp_emb_iq_range_penalty
            out_dict["stp_emb_iq_contour_penalty"] = stp_emb_iq_contour_penalty
            out_dict["stp_emb_iq_deviate_penalty"] = stp_emb_iq_deviate_penalty
            out_dict["stp_emb_iq_discrete_ppl"] = stp_emb_iq_discrete_ppl
            latents.append(stp_emb_iq_quantized)

            # This tensor converts discrete values to continuous.
            # It should *never* be used during training.
            out_dict["stp_emb_iq_quantized_lookup"] = tf.expand_dims(
                2. * (stp_emb_iq_discrete_f / (cfg.stp_emb_iq_nbins - 1.)) - 1., axis=2)


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
