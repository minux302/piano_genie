import numpy as np
import tensorflow.compat.v1 as tf
from magenta.music.protobuf import music_pb2

from config import Config


class SeqLoader():
    def __init__(self,
                 config,
                 repeat):
        self.batch_size = config.batch_size
        self.seq_len = config.seq_len
        self.max_discrete_times = config.max_discrete_times
        self.repeat = repeat
        self._build_pipeline(config.file_name)

    def _str_to_tensor(self, note_seq_str):
        note_seq = music_pb2.NoteSequence.FromString(note_seq_str)
        note_seq_ordered = sorted(list(note_seq.notes),
                                  key=lambda n: (n.start_time, n.pitch))
        note_seq_ordered = [n for n in note_seq_ordered
                            if (n.pitch >= 21) and (n.pitch <= 108)]

        pitches = np.array([note.pitch - 21 for note in note_seq_ordered])  # [21, 108) -> [0, 88)
        start_times = np.array([note.start_time for note in note_seq_ordered])
        # Delta time is a time between note on and next note on.
        if note_seq_ordered:
            # Delta time start hight to indicate free decision
            delta_times = np.concatenate([[100000.],
                                         start_times[1:] - start_times[:-1]])
        else:
            delta_times = np.zeros_like(start_times)

        return np.stack([pitches, delta_times], axis=1).astype(np.float32)

    def _filter_short(self, seq_tensor):
        seq_len = tf.shape(seq_tensor)[0]  # shape: (song_len, 2)
        return tf.greater_equal(seq_len, self.seq_len)

    def _random_crop(self, seq_tensor):
        seq_len = tf.shape(seq_tensor)[0]  # shape: (song_len, 2)
        start_max = seq_len - self.seq_len
        start_max = tf.maximum(start_max, 0)
        start = tf.random_uniform([], maxval=start_max + 1, dtype=tf.int32)
        cropped_seq_tensor = seq_tensor[start:start + self.seq_len]
        return cropped_seq_tensor

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
        self.iterator = dataset.make_one_shot_iterator()

    def _delta_times_to_int(self, delta_times):
        # Onsets and frames model samples at 31.25Hz
        delta_times_int = tf.cast(
            tf.round(delta_times * 31.25) + 1e-4, tf.int32)

        # Reduce time discretizations to a fixed number of buckets
        if self.max_discrete_times is not None:
            delta_times_int = tf.minimum(delta_times_int,
                                         self.max_discrete_times)
        return delta_times_int

    def get_batch(self):
        seq_tensors = self.iterator.get_next()
        seq_tensors.set_shape([self.batch_size, self.seq_len, 2])
        pitches = tf.cast(seq_tensors[:, :, 0] + 1e-4, tf.int32)
        delta_times = seq_tensors[:, :, 1]
        delta_times_int = self._delta_times_to_int(delta_times)
        return {"pitches": pitches,
                "delta_times_int": delta_times_int}


if __name__ == '__main__':
    config = Config()
    dataset = SeqLoader(config=config,
                        repeat=False)

    with tf.Session() as sess:
        for epoch in range(1):
            while True:
                try:
                    batch_data = sess.run(dataset.get_batch())
                    print(batch_data["pitches"])
                    print(batch_data["delta_times_int"])
                except tf.errors.OutOfRangeError:
                    break

    print('finished')
