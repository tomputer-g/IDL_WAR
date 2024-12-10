#!/usr/bin/env python3
import bchlib
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
import glob
from PIL import Image, ImageOps
import argparse
import models

# Import StegaStampRemover from your models.py
from models import StegaStampRemover

BCH_POLYNOMIAL = 137
BCH_BITS = 5

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/home/StegaStamp/saved_models/remover/model_50000.ckpt',
                        help='Path to StegaStampRemover checkpoint')
    parser.add_argument('--images_dir', type=str, default='./out/treering_ss/hidden',
                        help='Path to directory with input images')
    parser.add_argument('--output_dir', type=str, default='./out/treering_ss/remove',
                        help='Directory to save output images')
    parser.add_argument('--secret', type=str, default="Stega!!", 
                         help='Secret string to encode (max 7 characters)')
    parser.add_argument('--secret_size', type=int, default=100, help='Size of the secret vector')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Initialize TensorFlow session
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Image dimensions expected by the model
    height = 400
    width = 400
    secret_size = args.secret_size

    # Define placeholders
    remover_input_image_pl = tf.placeholder(shape=[None, height, width, 3], dtype=tf.float32, name="remover_input_image")
    secret_pl = tf.placeholder(shape=[None, secret_size], dtype=tf.float32, name="input_secret")

    # Build the StegaStampRemover model
    with tf.variable_scope('stega_stamp_remover_model'):
        remover = StegaStampRemover(secret_size=secret_size)
        remover_output = remover([remover_input_image_pl, secret_pl])

    # Collect variables to restore
    remover_vars = [var for var in tf.global_variables() if var.name.startswith('stega_stamp_remover_model/')]

    # Create saver and restore variables from checkpoint
    saver = tf.train.Saver(var_list=remover_vars)

    checkpoint_path = args.checkpoint  # Should be the common prefix, e.g., '/path/to/model_10000.ckpt'
    saver.restore(sess, checkpoint_path)

    # Making Secret Key
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    if len(args.secret) > 7:
        print('Error: Can only encode 56 bits (7 characters) with ECC')
        return

    data = bytearray(args.secret.ljust(7), 'utf-8')
    ecc = bch.encode(data)
    packet = data + ecc
    packet_binary = ''.join(format(byte, '08b') for byte in packet)
    secret_bits = [int(bit) for bit in packet_binary]
    secret_bits.extend([0] * (secret_size - len(secret_bits)))
    if len(secret_bits) != secret_size:
        print(f"Error: Secret bits length is {len(secret_bits)}, expected {secret_size}")
        return
    secret = np.array(secret_bits, dtype=np.float32)
    secret = np.expand_dims(secret, axis=0)  

    # Collect image files
    image_extensions = ('*.jpg', '*.jpeg', '*.png')
    files_list = []
    for ext in image_extensions:
        files_list.extend(glob.glob(os.path.join(args.images_dir, ext)))
    if not files_list:
        raise ValueError(f"No image files found in {args.images_dir}")

    # Process images
    for img_path in files_list:
        # Load and preprocess the image
        img = Image.open(img_path).convert("RGB")
        img = ImageOps.fit(img, (width, height))
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Create feed_dict
        feed_dict = {
            remover_input_image_pl: img,
            secret_pl: secret
        }

        # Run the model to get the output image
        output_img = sess.run(remover_output, feed_dict=feed_dict)

        # Postprocess and save the output image
        output_img = output_img[0]  # Remove batch dimension
        output_img = np.clip(output_img, 0.0, 1.0)
        output_img = (output_img * 255).astype(np.uint8)
        output_image_pil = Image.fromarray(output_img)

        # Save the output image
        img_name = os.path.basename(img_path)
        output_path = os.path.join(args.output_dir, img_name)
        output_image_pil.save(output_path)
        print(f"Processed and saved: {output_path}")

if __name__ == '__main__':
    main()
