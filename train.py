import tensorflow as tf
import numpy as np

import util
from loader import NoteSeqLoader
from model import PianoGenirModel


def train():
    dataset = NoteSeqLoader(file_name='midi_sample_tf',
                            batch_size=2,
                            seq_len=5,
                            repeat=False)

    model = PianoGenirModel(batch_size=1,
                            seq_len=5,
                            is_training=True)
    inputs = model.placeholders()
    outputs = model.build(inputs)

    with tf.Session() as sess:
        for epoch in range(1):
            sess.run(dataset.initializer())
            while True:
                try:
                    batch_data = sess.run(dataset.get_batch())
                    out = sess.run(outputs,
                                   feed_dict={pl: data
                                              for pl, data
                                              in zip(inputs, batch_data)})
                    print(np.array(out).shape)
                    # print(out)
                except tf.errors.OutOfRangeError:
                    break

    print('finished')


if __name__ == '__main__':
    train()
