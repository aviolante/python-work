# The MARCO data set contains 462,804 scored images from five source institutions:
# 0 - Collaborative Crystallisation Centre
# 1 - GlaxoSmithKline
# 2 - Hauptman-Woodward Medical Research Institute
# 3 - Merck & Co.
# 4 - Bristol-Myers Squibb

# Images Labels:
# 0 - Clear
# 1 - Crystals
# 2 - Other
# 3 - Precipitate

# TFRecord Features:
# image/height:        integer, image height in pixels
# image/width:         integer, image width in pixels
# image/colorspace:    string, specifying the colorspace, always 'RGB'
# image/channels:      integer, specifying the number of channels, always 3
# image/class/label:   integer, specifying the index in a normalized classification layer
# image/class/raw:     integer, specifying the index in the raw (original) classification layer
# image/class/source:  integer, specifying the index of the source (creator of the image)
# image/class/text:    string, specifying the human-readable version of the normalized label
# image/format:        string, specifying the format, always 'JPEG'
# image/filename:      string containing the basename of the image file
# image/id:            integer, specifying the unique id for the image
# image/encoded:       string, containing JPEG encoded image in RGB colorspace


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir('/Users/anviol/Desktop/Marco Project/train')

data_path = ['marcoTrainData00001.tfrecord',
             'marcoTrainData00002.tfrecord']

with tf.Session() as sess:
    feature = {'marcoTrainData00001/image//encoded': tf.FixedLenFeature([], tf.string),
               'marcoTrainData00001/image/class/label': tf.FixedLenFeature([], tf.int64),
               'marcoTrainData00002/image//encoded': tf.FixedLenFeature([], tf.string),
               'marcoTrainData00002/image/class/label': tf.FixedLenFeature([], tf.int64)}

    # Create list of file names and pass to queue
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)

    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)

    # Convert the image data from string back to numbers
    image = tf.decode_raw(features['marcoTrainData00001/image/encoded'], tf.float32)

    # Cast label data into int32
    label = tf.cast(features['marcoTrainData00001/image/class/label'])

    # Reshape the image data
    image = tf.reshape(image, [224, 224, 3])

    # Create batches of randomly shuffled tensors
    images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=5)

'''

Initialize and Plot Images

'''

# Initialize all global variables
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

# Create a coordinator and run all the QueueRunner objects
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

for batch_index in range(5):
    img, lbl = sess.run([images,labels])

    img = img.astype(np.uint8)

    for j in range(6):
        plt.subplot(2, 3, j+1)
        plt.imshow(img[j,...])
        plt.title('Clear' if (lbl[j]==0) else 'Not Clear')

    plt.show()

# Stop the threads
coord.request_stop()

# Wait for threads to stop
coord.join(threads)
sess.close()
