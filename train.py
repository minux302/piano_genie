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
    input_dict = model.placeholders()
    outputs = model.build(input_dict)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run([init_op])

        for epoch in range(1):
            sess.run(dataset.initializer())
            while True:
                try:
                    batch_datas = sess.run(dataset.get_batch())
                    out = sess.run(outputs,
                                   feed_dict={input_dict[key]: batch_datas[key]
                                              for key in input_dict.keys()})
                    print(np.array(out).shape)
                    # print(out)
                except tf.errors.OutOfRangeError:
                    break

    print('finished')


if __name__ == '__main__':
    train()
