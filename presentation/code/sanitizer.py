import collections
import tensorflow as tf
import dpsgd.utils as utils


ClipOption = collections.namedtuple("ClipOption",
                                    ["l2norm_bound", "clip"])


class AmortizedGaussianSanitizer(object):

  def __init__(self, accountant, default_option):

    self._accountant = accountant
    self._default_option = default_option
    self._options = {}

  def set_option(self, tensor_name, option):

    self._options[tensor_name] = option

  def sanitize(self, x, eps_delta, sigma=None,
               option=ClipOption(None, None), tensor_name=None,
               num_examples=None, add_noise=True):

    if sigma is None:
      eps, delta = eps_delta
      with tf.control_dependencies(
          [tf.Assert(tf.greater(eps, 0),
                     ["eps needs to be greater than 0"]),
           tf.Assert(tf.greater(delta, 0),
                     ["delta needs to be greater than 0"])]):
        sigma = tf.sqrt(2.0 * tf.math.log(1.25 / delta)) / eps

    l2norm_bound, clip = option
    if l2norm_bound is None:
      l2norm_bound, clip = self._default_option
      if ((tensor_name is not None) and
          (tensor_name in self._options)):
        l2norm_bound, clip = self._options[tensor_name]
    if clip:
      x = tf.clip_by_norm(x, clip_norm=l2norm_bound)
    
    if add_noise:
      if num_examples is None:
        num_examples = tf.slice(tf.shape(x), [0], [1])
      privacy_accum_op = self._accountant.accumulate_privacy_spending(
          eps_delta, sigma, num_examples)
      with tf.control_dependencies([privacy_accum_op]):
        saned_x = utils.AddGaussianNoise(x, sigma * l2norm_bound)
    else:
      saned_x = tf.reduce_sum(x, 0)
    return saned_x
