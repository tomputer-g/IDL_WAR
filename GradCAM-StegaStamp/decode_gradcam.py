import bchlib
import glob
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import cv2
import matplotlib.pyplot as plt

BCH_POLYNOMIAL = 137
BCH_BITS = 5

def compute_gradcam(sess, image, input_tensor, output_tensor, conv_layer):
    # Get the gradient of the output with respect to the conv layer
    # print(output_tensor)
    loss = tf.reduce_sum(output_tensor)
    grads = tf.gradients(loss, conv_layer)[0]

    printdebug = tf.print("Grads:", grads) # for debugging. Run the session against this node to print
    
    
    # Compute the weights (global average pooling of gradients)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    
    # Compute the weighted activation map
    cam = tf.reduce_sum(tf.multiply(weights, conv_layer), axis=-1)
    # print("Max: ", tf.maximum(cam,0))
    
    # Normalize the CAM
    cam = tf.maximum(cam, 0) / tf.reduce_max(cam)
    
    # Run the session to get the CAM
    cam_value = sess.run(cam, feed_dict={input_tensor: [image]})

    # Run the print
    # sess.run(printdebug, feed_dict={input_tensor: [image]})
    
    return cam_value[0]

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

    #The model ends with a sigmoid and then a rounding operation. The rounding destroys the gradient information and makes it return None.
    #Instead we get the output of any layer before rounding and use it to compute the gradient
    output_secret_sigmoid_name = 'Sigmoid:0' #'stega_stamp_decoder_1/sequential_1/dense_3/BiasAdd:0' 
    output_secret_sigmoid = tf.get_default_graph().get_tensor_by_name(output_secret_sigmoid_name)

    # # Iterate through operations and print output tensor names (for debugging and picking our output layer)
    # for op in tf.get_default_graph().get_operations():
    #     print(op.name)
    #     for output in op.outputs:
    #         print(" ", output.name)

    conv_layers = [op for op in sess.graph.get_operations() if op.type == 'Conv2D']
    last_conv_layer = conv_layers[-1].outputs[0]
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    for filename in files_list:
        image = Image.open(filename).convert("RGB")
        image = np.array(ImageOps.fit(image,(400, 400)),dtype=np.float32)
        image /= 255.
        feed_dict = {input_image:[image]}
        # ---------------------------------Decode StegaStamp------------------------------------
        secret = sess.run([output_secret],feed_dict=feed_dict)[0][0]
        packet_binary = "".join([str(int(bit)) for bit in secret[:96]])
        packet = bytes(int(packet_binary[i : i + 8], 2) for i in range(0, len(packet_binary), 8))
        packet = bytearray(packet)
        data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]
        bitflips = bch.decode_inplace(data, ecc)
        # ----------------------------(Separately) Compute GradCAM-------------------------------
        cam = compute_gradcam(sess, image, input_image, output_secret_sigmoid, last_conv_layer)
        # Resize CAM to match input image size
        cam_resized = cv2.resize(cam, (400, 400))

        # Apply CAM as heatmap to original image
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        # superimposed_img = cv2.addWeighted(np.uint8(image*255), 0.6, heatmap, 0.4, 0)

        # ----------------------------------Display GradCAM------------------------------------
        # Display the original image and GradCAM output
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
        plt.title("GradCAM Output")
        plt.axis('off')
        plt.show()
        # ---------------------------------Go to next image------------------------------------
        if bitflips != -1:
            try:
                code = data.decode("utf-8")
                print(filename, code)
                continue
            except:
                continue
        print(filename, 'Failed to decode')


if __name__ == "__main__":
    main()
