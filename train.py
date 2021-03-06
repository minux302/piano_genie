import tensorflow as tf
import numpy as np

from loader import SeqLoader
from model import PianoGenieModel
from config import Config


def train():

    config = Config()
    dataset = SeqLoader(config=config, repeat=False)

    model = PianoGenieModel(config=config, is_training=True)
    input_dict = model.placeholders()
    outputs = model.build(input_dict)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run([init_op])

        for epoch in range(1):
            while True:
                try:
                    batch_datas = sess.run(dataset.get_batch())
                    out = sess.run(outputs,
                                   feed_dict={input_dict[key]: batch_datas[key]
                                              for key in input_dict.keys()})
                    # print(np.array(out).shape)
                    print(out["dec_outputs"].shape)
                except tf.errors.OutOfRangeError:
                    break

    print('finished')


if __name__ == '__main__':
    train()
