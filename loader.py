import sys

import numpy as np
import tensorflow.compat.v1 as tf
from magenta.music.protobuf import music_pb2


class NoteSeqLoader():
    def __init__(self,
                 file_name,
                 batch_size,
                 seq_len,
                 repeat):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.repeat = repeat
        self._build_pipeline(file_name)

    def _str_to_tensor(self, note_sequence_str):
        note_sequence = music_pb2.NoteSequence.FromString(note_sequence_str)
        note_sequence_ordered = list(note_sequence.notes)
        note_sequence_ordered = sorted(note_sequence_ordered,
                                       key=lambda n: (n.start_time, n.pitch))
        note_sequence_ordered = [n for n in note_sequence_ordered
                                 if (n.pitch >= 21) and (n.pitch <= 108)]

        pitches = np.array([note.pitch for note in note_sequence_ordered])
        start_times = np.array([note.start_time for note in note_sequence_ordered])
        end_times = np.array([note.end_time for note in note_sequence_ordered])
        # Delta time is a time between note on and next note on.
        if note_sequence_ordered:
            # Delta time start hight to indicate free decision
            delta_times = np.concatenate([[100000.],
                                         start_times[1:] - start_times[:-1]])
        else:
            delta_times = np.zeros_like(start_times)

        return np.stack([pitches, delta_times, start_times, end_times],
                        axis=1).astype(np.float32)

    def _filter_short(self, note_sequence_tensor):
        note_sequence_len = tf.shape(note_sequence_tensor)[0]  # shape: (song_len, 4)
        return tf.greater_equal(note_sequence_len, self.seq_len)

    def _random_crop(self, note_sequence_tensor):
        note_sequence_len = tf.shape(note_sequence_tensor)[0]  # shape: (song_len, 4)
        start_max = note_sequence_len - self.seq_len
        start_max = tf.maximum(start_max, 0)
        start = tf.random_uniform([], maxval=start_max + 1, dtype=tf.int32)
        seq = note_sequence_tensor[start:start + self.seq_len]
        return seq

    def _build_pipeline(self, file_name):
        dataset = tf.data.TFRecordDataset(file_name)
        dataset = dataset.map(
            lambda data: tf.py_func(
                self._str_to_tensor,
                [data],
                [tf.float32],
                stateful=False))

        # Filter sequences that are too short
        dataset = dataset.filter(self._filter_short)
        # Get random seq_len crops from songs
        dataset = dataset.map(self._random_crop)

        if self.repeat:
            dataset = dataset.shuffle(buffer_size=512)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        self.iterator = dataset.make_initializable_iterator()

    def initializer(self):
        return self.iterator.initializer

    def get_batch(self):
        note_sequence_tensors = self.iterator.get_next()
        note_sequence_tensors.set_shape([self.batch_size, self.seq_len, 4])
        note_pitches = tf.cast(note_sequence_tensors[:, :, 0] + 1e-4, tf.int32)
        note_delta_times = note_sequence_tensors[:, :, 1]
        note_start_times = note_sequence_tensors[:, :, 2]
        note_end_times = note_sequence_tensors[:, :, 3]

        return {"pitches": note_pitches,
                "delta_times": note_delta_times,
                "start_times": note_start_times,
                "end_times": note_end_times}


if __name__ == '__main__':
    dataset = NoteSeqLoader(file_name='midi_sample_tf',
                            batch_size=1,
                            seq_len=5,
                            repeat=False)

    with tf.Session() as sess:
        for epoch in range(1):
            sess.run(dataset.initializer())
            while True:
                try:
                    batch_data = sess.run(dataset.get_batch())
                    # print(batch_data[0].shape)
                    # print(batch_data[1].shape)
                    # print(batch_data[2].shape)
                    # print(batch_data[3].shape)
                    print(batch_data[0])
                except tf.errors.OutOfRangeError:
                    break

    print('finished')
