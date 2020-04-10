import tensorflow as tf
import os

os.chdir('/Users/anviol/Desktop/train_test/')

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

# https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564

# Read and print data:
sess = tf.InteractiveSession()

# Read TFRecord file
reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(['marcoTrainData00001.tfrecord'])
_, serialized_example = reader.read(filename_queue)

# Define features
read_features = {
    'image/height': tf.FixedLenFeature([], dtype=tf.int64),
    'image/width': tf.FixedLenFeature([], dtype=tf.int64),
    'image/colorspace': tf.FixedLenFeature([], dtype=tf.string),
    'image/class/label': tf.FixedLenFeature([], dtype=tf.int64),
    'image/class/raw': tf.FixedLenFeature([], dtype=tf.int64),
    'image/class/source': tf.FixedLenFeature([], dtype=tf.int64),
    'image/class/text': tf.FixedLenFeature([], dtype=tf.string),
    'image/format': tf.FixedLenFeature([], dtype=tf.string),
    'image/filename': tf.FixedLenFeature([], dtype=tf.string),
    'image/id': tf.FixedLenFeature([], dtype=tf.int64),
    'image/encoded': tf.FixedLenFeature([], dtype=tf.string)
}

# Extract features from serialized data
read_data = tf.parse_single_example(serialized=serialized_example,
                                    features=read_features)

# Many tf.train functions use tf.train.QueueRunner,
# so we need to start it before we read
tf.train.start_queue_runners(sess)

# Print features
for name, tensor in read_data.items():
    print('{}: {}'.format(name, tensor.eval()))



