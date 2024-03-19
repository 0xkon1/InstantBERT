import os
import glob
import random
import tensorflow as tf

def split_tfrecord_files(tfrecord_dir, train_ratio=0.8):
    # Get all TFRecord files
    files = glob.glob(os.path.join(tfrecord_dir, '*.tfrecord'))
    
    # Shuffle files (optional)
    random.shuffle(files)

    # Split files into training and validation sets
    split_index = int(len(files) * train_ratio)
    train_files = files[:split_index]
    val_files = files[split_index:]

    return train_files, val_files

def copy_tfrecords(source_files, dest_dir):
    for file in source_files:
        # Define the destination file path
        dest_file_path = os.path.join(dest_dir, os.path.basename(file))
        
        with tf.io.TFRecordWriter(dest_file_path) as writer:
            for record in tf.data.TFRecordDataset(file):
                writer.write(record.numpy())

# Paths to your TFRecord directory and output directories for the split datasets
tfrecord_dir = './dataset/dataflow_output/'
train_dir = './dataset/train/'
val_dir = './dataset/validation/'

# Create output directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Split TFRecord files
train_files, val_files = split_tfrecord_files(tfrecord_dir)

# Copy records to new train and validation TFRecord files
copy_tfrecords(train_files, train_dir)
copy_tfrecords(val_files, val_dir)

print(f"Split {len(train_files)} files into training set, and {len(val_files)} files into validation set.")
