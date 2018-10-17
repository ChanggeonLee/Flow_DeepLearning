
# coding: utf-8

# In[1]:


import tensorflow as tf
import os
import numpy as np
from glob import glob as glb
from tqdm import *
import h5py
from tensorflow.contrib import rnn

min_queue_examples = 1000


# In[2]:


def load_flow(filename, shape):
  stream_flow = h5py.File(filename, 'r')
  flow_state_vel = np.array(stream_flow['Velocity_0'][:])
  flow_state_vel = flow_state_vel.reshape([shape[0], shape[1]+128, 3])[0:shape[0],0:shape[1],0:2]
  stream_flow.close()
  return flow_state_vel

def load_boundary(filename, shape):
  stream_boundary = h5py.File(filename, 'r')
  boundary_cond = np.array(stream_boundary['Gamma'][:])
  boundary_cond = boundary_cond.reshape([shape[0], shape[1]+128, 1])[0:shape[0],0:shape[1],:]
  stream_boundary.close()
  return boundary_cond

def read_data(filename_queue, shape):
  reader = tf.TFRecordReader()
  key, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example,
    features={
      'boundary':tf.FixedLenFeature([],tf.string),
      'sflow':tf.FixedLenFeature([],tf.string),
      'vmax':tf.FixedLenFeature([],tf.string),
    }) 
  boundary = tf.decode_raw(features['boundary'], tf.uint8)
  sflow = tf.decode_raw(features['sflow'], tf.float32)
  vmax = tf.decode_raw(features['vmax'], tf.float32)

  boundary = tf.reshape(boundary, [shape[0], shape[1], 1])
  sflow = tf.reshape(sflow, [shape[0], shape[1], 2])
  vmax = tf.reshape(vmax, [1024])
  boundary = tf.to_float(boundary)
  sflow = tf.to_float(sflow)
  vmax = tf.to_float(vmax)

  return boundary, sflow, vmax

def _generate_image_label_batch(boundary, sflow, vmax, batch_size, shuffle=True):
  num_preprocess_threads = 1
  #Create a queue that shuffles the examples, and then
  #read 'batch_size' images + labels from the example queue.
  boundarys, sflows, vmax = tf.train.shuffle_batch(
    [boundary, sflow, vmax],
    batch_size=batch_size,
    num_threads=num_preprocess_threads,
    capacity=min_queue_examples + 3 * batch_size,
    min_after_dequeue=min_queue_examples)
  return boundarys, sflows, vmax

def flow_inputs(batch_size):
  shape = (128,256)

  tfrecord_filename = glb('./data/*.tfrecords') 
  
  filename_queue = tf.train.string_input_producer(tfrecord_filename) 

  boundary, sflow,vmax = read_data(filename_queue, shape)

  boundarys, sflows, vmax = _generate_image_label_batch(boundary, sflow,vmax, batch_size)

  return boundarys, sflows, vmax


# In[3]:


n_batch = 8
learning_rate = 0.0001
keep_prob = 0.7
n_hidden = 128

global_step = tf.Variable(0, trainable=False, name='global_step')

X = tf.placeholder(tf.float32 , [None,128,256,1], name="X")

boundary, sflow, vmax = flow_inputs(n_batch)
X = boundary

v = tf.placeholder(tf.float32 ,[n_batch , 1024], name="v")
# v = tf.to_float(vmax)
v = vmax

print(v)

#Conv1
W1 = tf.Variable(tf.random_normal([16, 16, 1, 128], stddev=0.01))
L1 = tf.nn.conv2d(X, W1, strides=[1, 8, 16, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.dropout(L1, keep_prob)


# In[4]:


#Cov2
W2 = tf.Variable(tf.random_normal([4,4,128,512], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 4, 4, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.dropout(L2, keep_prob)


# In[5]:


L2_flat = tf.reshape(L2,[-1,4*4*512])


# In[6]:


W3 = tf.get_variable("W", shape=[512 * 4 * 4, 1024],
                     initializer=tf.contrib.layers.xavier_initializer())


# In[7]:


b3 = tf.Variable(tf.random_normal([1024]))
L3 = tf.nn.relu(tf.matmul(L2_flat, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

L3 = tf.add(L3 , v)


# In[8]:


# L3 = tf.divide(L3,y)
S1, S2 = tf.split(L3, [512, 512], 1)


# In[9]:


S1= tf.reshape(S1,[n_batch,1,1,512])
S2= tf.reshape(S2,[n_batch,1,1,512])


# In[10]:


# deconv1
W4_1 = tf.Variable(tf.random_normal([8,8,512,512], stddev=0.01))
L4_1 = tf.nn.conv2d_transpose(S1,W4_1,output_shape=[n_batch,8,8,512],strides=[1,8, 8, 1], padding='SAME')
L4_1 = tf.nn.relu(L4_1)
L4_1 = tf.nn.dropout(L4_1, keep_prob)

W4_2 = tf.Variable(tf.random_normal([8,8,512,512], stddev=0.01))
L4_2 = tf.nn.conv2d_transpose(S2,W4_2,output_shape=[n_batch,8,8,512],strides=[1,8, 8, 1], padding='SAME')
L4_2 = tf.nn.relu(L4_2)
L4_2 = tf.nn.dropout(L4_2, keep_prob)


# In[11]:


## deconv2
W5_1 = tf.Variable(tf.random_normal([4,8,256,512], stddev=0.01))
L5_1 = tf.nn.conv2d_transpose(L4_1,W5_1,output_shape=[n_batch,32,64,256],strides=[1, 4, 8, 1], padding='SAME')
L5_1 = tf.nn.relu(L5_1)
L5_1 = tf.nn.dropout(L5_1, keep_prob)

W5_2 = tf.Variable(tf.random_normal([4,8,256,512], stddev=0.01))
L5_2 = tf.nn.conv2d_transpose(L4_2,W5_2,output_shape=[n_batch,32,64,256],strides=[1,4, 8, 1], padding='SAME')
L5_2 = tf.nn.relu(L5_2)
L5_2 = tf.nn.dropout(L5_2, keep_prob)


# In[12]:


# deconv3
W6_1 = tf.Variable(tf.random_normal([2,2,32,256], stddev=0.01))
L6_1 = tf.nn.conv2d_transpose(L5_1,W6_1,output_shape=[n_batch,64,128,32],strides=[1,2, 2, 1], padding='SAME')
L6_1 = tf.nn.relu(L6_1)
L6_1 = tf.nn.dropout(L6_1, keep_prob)

W6_2 = tf.Variable(tf.random_normal([2,2,32,256], stddev=0.01))
L6_2 = tf.nn.conv2d_transpose(L5_2,W6_2,output_shape=[n_batch,64,128,32],strides=[1,2, 2, 1], padding='SAME')
L6_2 = tf.nn.relu(L6_2)
L6_2 = tf.nn.dropout(L6_2, keep_prob)


# In[13]:


# deconv4
W7_1 = tf.Variable(tf.random_normal([2,2,1,32], stddev=0.01))
L7_1 = tf.nn.conv2d_transpose(L6_1,W7_1,output_shape=[n_batch,128,256,1],strides=[1,2,2, 1], padding='SAME')
L7_1 = tf.nn.dropout(L7_1, keep_prob)

W7_2 = tf.Variable(tf.random_normal([2,2,1,32], stddev=0.01))
L7_2 = tf.nn.conv2d_transpose(L6_2,W7_2,output_shape=[n_batch,128,256,1],strides=[1,2, 2, 1], padding='SAME')
L7_2 = tf.nn.dropout(L7_2, keep_prob)


# In[14]:


sflow_p = tf.stack([L7_1 , L7_2] , axis=3)
sflow_p = tf.reshape(sflow_p , [n_batch,128,256,2])
# loss = tf.reduce_mean(tf.square(sflow_p - sflow))
loss = tf.nn.l2_loss(sflow_p - sflow)


# In[15]:


#loss 
total_loss = tf.train.AdamOptimizer(learning_rate).minimize(loss)


# In[16]:


sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())
ckpt = tf.train.get_checkpoint_state('./model_save')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())


# In[17]:


tf.train.start_queue_runners(sess=sess)


# In[18]:


# v = sess.run([v],feed_dict={})
# print(v)


# In[20]:


steps = 10000

min_cost = 10000
for epoch in range(steps):
  total_cost = 0
  _, cost_val = sess.run([total_loss, loss],feed_dict={})
  total_cost += cost_val
  if min_cost > total_cost:
    print("saver point")
    min_cost = total_cost
    saver.save(sess, "./model_save/model.ckpt")

  print('Epoch:', '%04d' % (epoch + 1),
        'Avg. cost =', '{:f}'.format(total_cost))
print("최적화 완료")

