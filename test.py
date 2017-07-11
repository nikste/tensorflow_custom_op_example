#!/usr/bin/env python

import tensorflow as tf


from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print('available device')
print(get_available_gpus())


print('instantiating')

zero_out_module = tf.load_op_library('./zero_out.so')

with tf.device('/cpu:0'):
 ret_cpu = zero_out_module.zero_out([[1, 2], [3, 4]])

# with tf.device('/gpu:0'):
#   ret_gpu = zero_out_module.zero_out([[1, 2], [3, 4]])

with tf.Session(config=tf.ConfigProto(
                log_device_placement=True
                # allow_soft_placement=True
            )) as sess:

  # print('running gpu')
  # print(sess.run(ret_gpu))

  # cpu
  # print('..')
  print('running cpu')
  print(sess.run(ret_cpu))

# print(ret)
exit(0)
