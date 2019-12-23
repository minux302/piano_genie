import numpy as np
import tensorflow.compat.v1 as tf


def demidity(pitches):
  """Transforms MIDI pitches [21,108] to [0, 88)."""
  assertions = [
      tf.assert_greater_equal(pitches, 21),
      tf.assert_less_equal(pitches, 108)
  ]
  with tf.control_dependencies(assertions):
    return pitches - 21


def remidify(pitches):
  """Transforms [0, 88) to MIDI pitches [21, 108]."""
  assertions = [
      tf.assert_greater_equal(pitches, 0),
      tf.assert_less_equal(pitches, 87)
  ]
  with tf.control_dependencies(assertions):
    return pitches + 21


# def discrete_to_piano_roll(categorical, dim, dilation=1, colorize=True):
