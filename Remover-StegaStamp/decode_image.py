import bchlib
import glob
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import tensorflow.contrib.image
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tqdm import tqdm

from models import get_secret_acc

BCH_POLYNOMIAL = 137
BCH_BITS = 5

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--images_dir', type=str, default=None)
    parser.add_argument('--secret_size', type=int, default=100)
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

    # For calculating the bit accuracy
    secret_key = 'Stega!!'
    data_gt = bytearray(secret_key + ' ' * (7 - len(secret_key)), 'utf-8')
    ecc_gt = bch.encode(data_gt)
    packet_gt = data_gt + ecc_gt

    packet_gt_binary = ''.join(format(x, '08b') for x in packet_gt)
    secret_gt = [int(x) for x in packet_gt_binary]
    secret_gt.extend([0,0,0,0])

    acc = 0
    bit_acc = 0
    detection_rate = 0

    for filename in tqdm(files_list):
        image = Image.open(filename).convert("RGB")
        image = np.array(ImageOps.fit(image,(400, 400)),dtype=np.float32)
        image /= 255.

        feed_dict = {input_image:[image]}

        secret = sess.run([output_secret],feed_dict=feed_dict)
        secret = secret[0]

        bit_acc_temp = get_secret_acc(secret_gt, secret)
        # print(bit_acc_temp[0].eval())
        bit_acc += bit_acc_temp[0].eval() 

        packet_binary = "".join([str(int(bit)) for bit in secret[0][:96]])
        packet = bytes(int(packet_binary[i : i + 8], 2) for i in range(0, len(packet_binary), 8))
        packet = bytearray(packet)

        data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]

        bitflips = bch.decode_inplace(data, ecc)

        if bitflips != -1:
            try:
                code = data.decode("utf-8")
                detection_rate += 1
                if code == secret_key:
                    acc += 1
                continue
            except:
                continue
        # print(filename, 'Failed to decode')
    
    print(f"Bit Accuracy: {bit_acc/len(files_list)}")
    print(f"Accuracy: {acc/len(files_list)}")
    print(f"Detection Rate: {detection_rate/len(files_list)}")

if __name__ == "__main__":
    main()
