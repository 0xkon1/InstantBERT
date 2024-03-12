import tensorflow as tf

# Replace this with the path to your TFRecord file
file_path = 'dataset/dataflow_output/train-00100-of-01000.tfrecord'

# Create a dataset from the TFRecord file
dataset = tf.data.TFRecordDataset(file_path)

# Iterate over the dataset
for raw_record in dataset:
    # Parse each raw record into an `Example` object
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    
    # Print the `Example`
    print(example)