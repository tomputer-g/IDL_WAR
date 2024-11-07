import tensorflow as tf
import numpy as np
import os
import glob
from PIL import Image, ImageOps
import random
from tensorflow.python.saved_model import tag_constants, signature_constants

# Paths
ENCODER_MODEL_PATH = '/ocean/projects/cis220031p/sbaek/StegaStamp/StegaStamp/saved_models/original'
REMOVER_CHECKPOINTS_PATH = '/ocean/projects/cis220031p/sbaek/StegaStamp/StegaStamp/remover_checkpoints/'
if not os.path.exists(REMOVER_CHECKPOINTS_PATH):
    os.makedirs(REMOVER_CHECKPOINTS_PATH)

TRAIN_PATH = '/ocean/projects/cis220031p/sbaek/StegaStamp/image/coco2017/train2017/'

def get_img_batch(files_list, batch_size=4, size=(400, 400)):
    batch_images = []
    for _ in range(batch_size):
        img_path = random.choice(files_list)
        try:
            img = Image.open(img_path).convert("RGB")
            img = ImageOps.fit(img, size)
            img = np.array(img, dtype=np.float32) / 255.0
        except:
            img = np.zeros((size[0], size[1], 3), dtype=np.float32)
        batch_images.append(img)
    return np.array(batch_images)

def generate_secrets(secret_size, batch_size):
    # Generate random binary secrets for each item in the batch
    secrets = np.random.binomial(1, 0.5, (batch_size, secret_size))
    return secrets.astype(np.float32)

def get_watermarked_batch(sess, input_secret, input_image, output_residual, files_list, secret_size, batch_size=4, size=(400, 400)):
    # Generate watermarked images using the trained encoder
    batch_cover = get_img_batch(files_list, batch_size, size)
    batch_secret = generate_secrets(secret_size, batch_size)
    feed_dict = {input_secret: batch_secret, input_image: batch_cover}
    residual = sess.run(output_residual, feed_dict=feed_dict)
    # Compute encoded images using residual
    encoded_images = batch_cover + residual
    encoded_images = np.clip(encoded_images, 0, 1)
    return batch_cover, encoded_images

def train_watermark_remover(files_list, secret_size, args):
    tf.reset_default_graph()

    # Placeholders for images
    clean_image_pl = tf.placeholder(tf.float32, shape=[None, 400, 400, 3], name="clean_image_input")
    watermarked_image_pl = tf.placeholder(tf.float32, shape=[None, 400, 400, 3], name="watermarked_image_input")

    # Define the remover model
    with tf.variable_scope('remover'):
        x = watermarked_image_pl - 0.5
        x = tf.layers.conv2d(x, 32, (3, 3), activation=tf.nn.relu, padding='same')
        x = tf.layers.conv2d(x, 64, (3, 3), activation=tf.nn.relu, padding='same')
        x = tf.layers.conv2d(x, 128, (3, 3), activation=tf.nn.relu, padding='same')
        x = tf.layers.conv2d(x, 64, (3, 3), activation=tf.nn.relu, padding='same')
        x = tf.layers.conv2d(x, 32, (3, 3), activation=tf.nn.relu, padding='same')
        restored_image = tf.layers.conv2d(x, 3, (1, 1), activation=None, padding='same') + 0.5

    # Define the removal loss
    removal_loss = tf.reduce_mean(tf.square(restored_image - clean_image_pl))
    optimizer = tf.train.AdamOptimizer(args.lr).minimize(removal_loss)

    # TensorBoard summaries
    summary_op = tf.summary.merge([tf.summary.scalar('removal_loss', removal_loss)])

    # Saver for the remover model
    remover_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='remover')
    remover_saver = tf.train.Saver(var_list=remover_vars, max_to_keep=10)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Load the encoder model
        model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], ENCODER_MODEL_PATH)

        # Access encoder's input and output tensors using the SavedModel signature
        input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
        input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
        output_residual_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['residual'].name

        input_secret = tf.get_default_graph().get_tensor_by_name(input_secret_name)
        input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)
        output_residual = tf.get_default_graph().get_tensor_by_name(output_residual_name)

        # TensorBoard writer
        log_dir = os.path.join(REMOVER_CHECKPOINTS_PATH, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = tf.summary.FileWriter(log_dir, sess.graph)

        global_step = 0
        total_steps = args.num_steps

        for step in range(total_steps):
            # Generate watermarked images using the trained encoder
            clean_images, watermarked_images = get_watermarked_batch(
                sess, input_secret, input_image, output_residual,
                files_list=files_list,
                secret_size=secret_size,
                batch_size=args.batch_size
            )

            # Training step
            feed_dict = {clean_image_pl: clean_images, watermarked_image_pl: watermarked_images}
            _, loss, summary = sess.run([optimizer, removal_loss, summary_op], feed_dict)

            # Log summaries every 100 steps
            if global_step % 100 == 0:
                writer.add_summary(summary, global_step)
                writer.flush()
                print(f"Step {global_step}: Removal loss = {loss:.6f}")

            # Save checkpoint every 1000 steps
            if global_step % 1000 == 0:
                remover_saver.save(sess, os.path.join(REMOVER_CHECKPOINTS_PATH, f'remover_model_step_{global_step}.ckpt'))
            global_step += 1

        writer.close()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_steps', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--secret_size', type=int, default=100)  # Adjust based on your secret size
    args = parser.parse_args()

    # Collect image files
    image_extensions = ('*.jpg', '*.jpeg', '*.png')
    files_list = []
    for ext in image_extensions:
        files_list.extend(glob.glob(os.path.join(TRAIN_PATH, "**", ext), recursive=True))

    if not files_list:
        raise ValueError(f"No image files found in {TRAIN_PATH}")

    # Start training the watermark remover
    train_watermark_remover(files_list, args.secret_size, args)

if __name__ == "__main__":
    main()
