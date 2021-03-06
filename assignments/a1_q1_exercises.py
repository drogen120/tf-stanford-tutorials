"""
Simple TensorFlow exercises
You should thoroughly test your code
"""

import tensorflow as tf

###############################################################################
# 1a: Create two random 0-d tensors x and y of any distribution.
# Create a TensorFlow object that returns x + y if x > y, and x - y otherwise.
# Hint: look up tf.cond()
# I do the first problem for you
###############################################################################

x = tf.random_uniform([])  # Empty array as shape creates a scalar.
y = tf.random_uniform([])
out = tf.cond(tf.less(x, y), lambda: tf.add(x, y), lambda: tf.sub(x, y))

###############################################################################
# 1b: Create two 0-d tensors x and y randomly selected from -1 and 1.
# Return x + y if x < y, x - y if x > y, 0 otherwise.
# Hint: Look up tf.case().
###############################################################################

# YOUR CODE
x = tf.random_uniform([], minval = -1, maxval = 1, dtype = tf.float32)
y = tf.random_uniform([], minval = -1, maxval = 1, dtype = tf.float32)

f1 = lambda: tf.add(x, y)
f2 = lambda: tf.sub(x, y)
f3 = lambda: tf.constant(0.0)
out = tf.case({tf.less(x, y) : f1, tf.greater(x, y) : f2}, default = f3)
print out

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print sess.run(out)

###############################################################################
# 1c: Create the tensor x of the value [[0, -2, -1], [0, 1, 2]]
# and y as a tensor of zeros with the same shape as x.
# Return a boolean tensor that yields Trues if x equals y element-wise.
# Hint: Look up tf.equal().
###############################################################################

# YOUR CODE
x = tf.Variable([[0, -2, -1], [0, 1, 2]])
y = tf.zeros(x.get_shape(), dtype = tf.int32)

out = tf.equal(x, y)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print sess.run(out)

###############################################################################
# 1d: Create the tensor x of value
# [29.05088806,  27.61298943,  31.19073486,  29.35532951,
#  30.97266006,  26.67541885,  38.08450317,  20.74983215,
#  34.94445419,  34.45999146,  29.06485367,  36.01657104,
#  27.88236427,  20.56035233,  30.20379066,  29.51215172,
#  33.71149445,  28.59134293,  36.05556488,  28.66994858].
# Get the indices of elements in x whose values are greater than 30.
# Hint: Use tf.where().
# Then extract elements whose values are greater than 30.
# Hint: Use tf.gather().
###############################################################################

# YOUR CODE
#x = tf.random_uniform([20], minval = 28, maxval = 32, dtype = tf.float32)
x = tf.Variable([29.05088806,  27.61298943,  31.19073486,  29.35532951, 30.97266006,  26.67541885,  38.08450317,  20.74983215, \
                 34.94445419,  34.45999146,  29.06485367,  36.01657104, 27.88236427,  20.56035233,  30.20379066,  29.51215172, \
                 33.71149445,  28.59134293,  36.05556488,  28.66994858])
y = tf.constant(30.0, dtype = tf.float32)
index_matrix = tf.where(tf.greater(x, y))
output = tf.gather(x, index_matrix)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print sess.run(x)
    print sess.run(index_matrix)
    print sess.run(output)

###############################################################################
# 1e: Create a diagnoal 2-d tensor of size 6 x 6 with the diagonal values of 1,
# 2, ..., 6
# Hint: Use tf.range() and tf.diag().
###############################################################################

# YOUR CODE
dia = tf.range(1, 7, 1)
matrix = tf.diag(dia)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print sess.run(matrix)

###############################################################################
# 1f: Create a random 2-d tensor of size 10 x 10 from any distribution.
# Calculate its determinant.
# Hint: Look at tf.matrix_determinant().
###############################################################################

# YOUR CODE
x = tf.random_uniform([2,2], minval = 1, maxval = 10, dtype = tf.float32)
det = tf.matrix_determinant(x)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    xx, dett = sess.run([x,det])
    print xx
    print dett

###############################################################################
# 1g: Create tensor x with value [5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9].
# Return the unique elements in x
# Hint: use tf.unique(). Keep in mind that tf.unique() returns a tuple.
###############################################################################

# YOUR CODE
x = tf.Variable([5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9])
uniq = tf.unique(x)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print sess.run(uniq)

###############################################################################
# 1h: Create two tensors x and y of shape 300 from any normal distribution,
# as long as they are from the same distribution.
# Use tf.less() and tf.select() to return:
# - The mean squared error of (x - y) if the average of all elements in (x - y)
#   is negative, or
# - The sum of absolute value of all elements in the tensor (x - y) otherwise.
# Hint: see the Huber loss function in the lecture slides 3.
###############################################################################

# YOUR CODE
