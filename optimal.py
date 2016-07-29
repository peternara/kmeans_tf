"""
KMEANS using Tensorflow with Input File (Optimal Version)
Francesc Sastre Cabot, github: xiscosastre
Base code by Jordi Torres, github: jorditorresBCN
Usage example:
python script.py dataset_file num_clusters num_steps


Input file format
id1 k1 k2 k3 ... kn
id2 k1 k2 k3 ... kn
id3 k1 k2 k3 ... kn
...
idn k1 k2 k3 ... kn
"""

import numpy as np
import tensorflow as tf
import sys
import time

num_clusters = int(sys.argv[2])
num_steps = int(sys.argv[3])
print("Kmeans (Optimal Version) with " + str(num_clusters) + " clusters and " + str(num_steps) + " steps")
begin_io_time = time.time()
# Read input_file
vector_values = np.loadtxt(sys.argv[1])
# Delete first row (id)
vector_values = np.delete(vector_values, 0, 1)
print("Total IO Time: %3.2fs" % float(time.time() - begin_io_time))

vectors = tf.constant(vector_values)
centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors),
                                 [0, 0], [num_clusters, -1]))
expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroids = tf.expand_dims(centroids, 1)

distances = tf.reduce_sum(
    tf.square(tf.sub(expanded_vectors, expanded_centroids)), 2)
assignments = tf.argmin(distances, 0)

means = tf.concat(0, [
    tf.reduce_mean(
        tf.gather(vectors,
                  tf.reshape(
                      tf.where(
                          tf.equal(assignments, c)
                      ), [1, -1])
                  ), reduction_indices=[1])
    for c in range(num_clusters)])

update_centroids = tf.assign(centroids, means)
init_op = tf.initialize_all_variables()

# with tf.Session('local') as sess:
sess = tf.Session()
sess.run(init_op)

begin_time = time.time()
for step in range(num_steps):
    _, centroid_values, assignment_values = sess.run([update_centroids,
                                                      centroids,
                                                      assignments])
print("Total Ex Time: %3.2fs" % float(time.time() - begin_time))

print ("Centroids: " + str(centroid_values))

