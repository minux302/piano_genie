import tensorflow as tf
import numpy as np

from loader import NoteSeqLoader
from model import PianoGenirModel
from config import Config


def train():

    config = Config()
    dataset = NoteSeqLoader(file_name=config.file_name,
                            batch_size=config.batch_size,
                            seq_len=config.seq_len,
                            repeat=False)

    model = PianoGenirModel(config=config,
                            is_training=True)
    input_pls = model.placeholders()
    outputs = model.build(input_pls)

    with tf.Session() as sess:
        for epoch in range(1):
            sess.run(dataset.initializer())
            while True:
                try:
                    batch_datas = sess.run(dataset.get_batch())
                    out = sess.run(outputs,
                                   feed_dict={input_pl: batch_data
                                              for input_pl, batch_data
                                              in zip(input_pls, batch_datas)})
                    print(np.array(out).shape)
                    # print(out)
                except tf.errors.OutOfRangeError:
                    break

    print('finished')


if __name__ == '__main__':
    train()
