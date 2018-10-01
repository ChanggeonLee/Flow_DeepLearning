import tensorflow as tf

# option setting
learning_rate = 0.01
training_epoch = 3000
# batch_size =
 
shape = [128, 256]

X = tf.placeholder(tf.float32 , [None, 256, 128, 1])
Y = tf.placeholder(tf.float32 , [None, 256, 128, 1])
is_training = tf.placeholder(tf.bool)

# CNN encoding layer 
# Conv
L1 = tf.layers.conv2d(X, 128,[16,8])
L1 = tf.layers.max_pooling2d(L1, [16,16], [16,16])
L1 = tf.layers.dropout(L1 , 0.7, is_training)

# Conv
L2 = tf.layers.conv2d(L1, 512,[4,4])
L2 = tf.layers.max_pooling2d(L2, [4,4], [4,4])
L2 = tf.layers.dropout(L2 , 0.7, is_training)

# FullCon
W4 = tf.Variable(tf.random_normal([256, 1024], stddev=0.01))
print(W4)

# CNN decoding layer


# loss


# sess run