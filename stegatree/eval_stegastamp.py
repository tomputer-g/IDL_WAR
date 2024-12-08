import bchlib
import glob
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import tensorflow.contrib.image
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from PIL import Image, ImageFilter
from tqdm import tqdm

BCH_POLYNOMIAL = 137
BCH_BITS = 5

def identity(img):
    return img

def rotation_attack(img):
    return img.rotate(75)

def blur_attack(img):
    return img.filter(ImageFilter.GaussianBlur(radius=4))

def get_attack(attack_name):
    if attack_name == "none":
        return identity
    if attack_name == "rotation":
        return rotation_attack
    if attack_name == "blur":
        return blur_attack
    
    raise Exception(f"Unimplemented attack {attack_name} requested")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--images_dir', type=str, default=None)
    parser.add_argument('--secret_size', type=int, default=100)
    parser.add_argument('--output_file', type=str, default="stegastamp_eval.csv")
    parser.add_argument('--attack', type=str, default='none')
    args = parser.parse_args()

    if args.attack != "none":
        attack_name = args.attack
    else:
        attack_name = ""
    attack = get_attack(args.attack)

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

    with open(attack_name+"_"+args.output_file, mode="w") as f:
        for filename in tqdm(files_list):
            image = Image.open(filename).convert("RGB")
            image = attack(image)
            image = np.array(ImageOps.fit(image,(400, 400)),dtype=np.float32)
            image /= 255.

            feed_dict = {input_image:[image]}

            secret = sess.run([output_secret],feed_dict=feed_dict)[0][0]

            packet_binary = "".join([str(int(bit)) for bit in secret[:96]])
            packet = bytes(int(packet_binary[i : i + 8], 2) for i in range(0, len(packet_binary), 8))
            packet = bytearray(packet)

            data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]

            bitflips = bch.decode_inplace(data, ecc)

            if bitflips != -1:
                try:
                    code = data.decode("utf-8")
                    f.write(f"{filename},true\n")
                    continue
                except:
                    f.write(f"{filename},true\n")
                    continue
            f.write(f"{filename},false\n")

    
if __name__ == "__main__":
    main()
