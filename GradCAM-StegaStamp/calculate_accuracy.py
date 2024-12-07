# Sourced from https://github.com/tancik/StegaStamp

import bchlib
import glob
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tqdm import tqdm
import gc
# import matplotlib.pyplot as plt

BCH_POLYNOMIAL = 137
BCH_BITS = 5

def get_secret_acc(secret_true,secret_pred):
    with tf.variable_scope("acc"):
        secret_pred = tf.round(tf.sigmoid(secret_pred))

        correct_pred = tf.cast(tf.shape(secret_pred)[1], dtype=tf.int64) - tf.count_nonzero(secret_pred - secret_true, axis=1)

        str_acc = 1.0 - tf.count_nonzero(correct_pred - tf.cast(tf.shape(secret_pred)[1], dtype=tf.int64)) / tf.size(correct_pred, out_type=tf.int64)

        bit_acc = tf.reduce_sum(correct_pred) / tf.size(secret_pred, out_type=tf.int64)
        return bit_acc, str_acc

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--images_dir', type=str, default=None)
    parser.add_argument('--secret_size', type=int, default=100)
    parser.add_argument('--secret_message', type=str, default="Stega!!")
    args = parser.parse_args()

    if args.image is not None:
        files_list = [args.image]
    elif args.images_dir is not None:
        files_list = glob.glob(args.images_dir + '/*')
    else:
        print('Missing input image')
        return

    sess = tf.InteractiveSession(graph=tf.Graph())

    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], args.model)

    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['decoded'].name
    output_secret = tf.get_default_graph().get_tensor_by_name(output_secret_name)

    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    data_gt = bytearray(args.secret_message + ' ' * (7 - len(args.secret_message)), 'utf-8')
    ecc_gt = bch.encode(data_gt)
    packet_gt = data_gt + ecc_gt

    packet_gt_binary = ''.join(format(x, '08b') for x in packet_gt)
    secret_gt = [int(x) for x in packet_gt_binary]
    secret_gt.extend([0] * (args.secret_size - len(secret_gt)))

    bit_acc = 0
    detections = 0

    for filename in tqdm(files_list):
        image = Image.open(filename).convert("RGB")
        image = np.array(ImageOps.fit(image,(400, 400)),dtype=np.float32)
        image /= 255.
        feed_dict = {input_image:[image]}
        # ---------------------------------Decode StegaStamp------------------------------------
        secret = sess.run([output_secret],feed_dict=feed_dict)[0]
        # bit_acc_temp = get_secret_acc(secret_gt, secret)
        # bit_acc += bit_acc_temp[0].eval()

        packet_binary = "".join([str(int(bit)) for bit in secret[0][:96]])
        packet = bytes(int(packet_binary[i : i + 8], 2) for i in range(0, len(packet_binary), 8))
        packet = bytearray(packet)
        data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]
        bitflips = bch.decode_inplace(data, ecc)
        
        # ---------------------------------Go to next image------------------------------------
        gc.collect()
        
        if bitflips != -1:
            try:
                code = data.decode("utf-8")
                # print(filename, code)
                detections += 1
                continue
            except:
                continue
        # print(filename, 'Failed to decode')

    print(f"Detection Rate: {detections/len(files_list)}")
    # print(f"Bit Accuracy: {bit_acc/len(files_list)}")

if __name__ == "__main__":
    main()
