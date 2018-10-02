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
# W1_0.01 = tf.Variable(tf.random_normal([3,3,1,16], stddev=0.01))
# W1_0.05 = tf.Variable(tf.random_normal([3,3,1,16], stddev=0.01))
# W1_0.1 = tf.Variable(tf.random_normal([3,3,1,16], stddev=0.01))

# Conv layer1
W1 = tf.Variable(tf.random_normal([3,3,1,16], stddev=0.01))
L1 = tf.nn.conv2d(X, W1, strides=1, padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pooling(L1, [1,2,2,1], [1,2,2,1], padding='SAME')

# Conv layer2
W2 = tf.Variable(tf.random_normal([3,3,1,16], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=1, padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pooling(L2, [1,2,2,1], [1,2,2,1], padding='SAME')


# FullC layer
W3 = tf.Variable(tf.random_normal([3,3,1,16], stddev=0.01))
L3 = tf.reshape(L2, [])
L3 = tf.matmul(L3,W3)
L3 = tf.nn.relu(L3)
L3 = tf.nn.dropout(L3, keep_prob)

W4 = tf.Variable(tf.random_normal([3,3,1,16], stddev=0.01))
model = tf.matmul(L3 , W4)

# Deconv layer1
# Deconv layer2
# Deconv layer3
# Deconv layer4

# gen flow

# loss 





L1 = tf.layers.conv2d(X, 128,[16,8])
L1 = tf.layers.max_pooling2d(L1, [16,16], [16,16])
L1 = tf.layers.dropout(L1 , 0.7, is_training)

# Conv
L2 = tf.layers.conv2d(L1, 512,[4,4])
L2 = tf.layers.max_pooling2d(L2, [4,4], [4,4])
L2 = tf.layers.dropout(L2 , 0.7, is_training)

# FullCon
W4 = tf.layers.flatten(L2)

print(W4)

# CNN decoding layer


# loss


# sess run