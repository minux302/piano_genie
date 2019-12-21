import random

import numpy as np
import tensorflow.compat.v1 as tf
from magenta.music.protobuf import music_pb2


def load_noteseqs(filename,
                  batch_size,
                  seq_len,
                  repeat=False):
    def _str_to_tensor(note_sequence_str):
        note_sequence = music_pb2.NoteSequence.FromString(note_sequence_str)
        note_sequence_ordered = list(note_sequence.notes)
        note_sequence_ordered = sorted(note_sequence_ordered,
                                       key=lambda n: (n.start_time, n.pitch))
        note_sequence_ordered = [n for n in note_sequence_ordered
                                 if (n.pitch >= 21) and (n.pitch <= 108)]

        pitches = np.array([note.pitch for note in note_sequence_ordered])
        velocities = np.array([note.velocity for note in note_sequence_ordered])
        start_times = np.array([note.start_time for note in note_sequence_ordered])
        end_times = np.array([note.end_time for note in note_sequence_ordered])

        # Todo, what is delta_times ?
        if note_sequence_ordered:
            # Delta time start hight to indicate free decision
            delta_times = np.concatenate([[100000.],
                                         start_times[1:] - start_times[:-1]])
        else:
            delta_times = np.zeros_like(start_times)

        return note_sequence_str, np.stack(
            [pitches, velocities, delta_times, start_times, end_times],
            axis=1).astype(np.float32)

    def _filter_short(note_sequence_tensor, seq_len):
        note_sequence_len = tf.shape(note_sequence_tensor)[0]
        return tf.greater_equal(note_sequence_len, seq_len)

    def _random_crop(note_sequence_tensor, seq_len):
        note_sequence_len = tf.shape(note_sequence_tensor)[0]
        start_max = note_sequence_len - seq_len
        start_max = tf.maximum(start_max, 0)

        start = tf.random_uniform([], maxval=start_max + 1, dtype=tf.int32)
        seq = note_sequence_tensor[start:start + seq_len]

        return seq

    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(
        lambda data: tf.py_func(
            lambda x: _str_to_tensor(x),
            [data],
            (tf.string, tf.float32),
            stateful=False))

    # Filter sequences that are too short
    dataset = dataset.filter(lambda s, n: _filter_short(n, seq_len))

    # Get random crops
    # dataset = dataset.map(lambda s, n: (s, _random_crop(n, seq_len)))

    # Shuffle
    if repeat:
        dataset = dataset.shuffle(buffer_size=512)

    # Make batches
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Get tensors
    iterator = dataset.make_one_shot_iterator()  # Todo: why here is one_shot iterator ?
    note_sequence_strs, note_sequence_tensors = iterator.get_next()

    # Set shapes
    # why here is need ?
    note_sequence_strs.set_shape([batch_size])
    note_sequence_tensors.set_shape([batch_size, seq_len, 5])

    # Retrieve tensors
    note_pitches = tf.cast(note_sequence_tensors[:, :, 0] + 1e-4, tf.int32)
    note_velocities = tf.cast(note_sequence_tensors[:, :, 1] + 1e-4, tf.int32)
    note_delta_times = note_sequence_tensors[:, :, 2]
    note_start_times = note_sequence_tensors[:, :, 3]
    note_end_times = note_sequence_tensors[:, :, 4]

    # Onsets and frames model samples at 31.25Hz
    note_delta_times_int = tf.cast(
        tf.round(note_delta_times * 31.25) + 1e-4, tf.int32)

    # Build return dict
    note_tensors = {
        "pb_strs": note_sequence_strs,
        "midi_pitches": note_pitches,
        "velocities": note_velocities,
        "delta_times": note_delta_times,
        "delta_times_int": note_delta_times_int,
        "start_times": note_start_times,
        "end_times": note_end_times
    }

    return note_tensors


if __name__ == '__main__':
    # Load data
    with tf.name_scope("loader"):
        feat_dict = load_noteseqs(filename='midi_sample_tf',
                                  batch_size=1,
                                  seq_len=10,
                                  repeat=False)

    with tf.Session() as sess:
        for epoch in range(1):
            while True:
                try:
                    batch = sess.run(feat_dict)
                    print(batch['midi_pitches'])
                except tf.errors.OutOfRangeError:
                    break
    print('finished')
