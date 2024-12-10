#!/usr/bin/env python3
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
import glob
from PIL import Image, ImageOps
import random
from tensorflow.python.saved_model import tag_constants, signature_constants
from tqdm import tqdm  # tqdm 임포트

import lpips.lpips_tf as lpips_tf
import utils
from tensorflow import keras
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from models import StegaStampRemover
import models

# Paths
ENCODER_MODEL_PATH = '/home/StegaStamp/saved_models/test/'
CHECKPOINTS_PATH = './saved_models/'
if not os.path.exists(CHECKPOINTS_PATH):
    os.makedirs(CHECKPOINTS_PATH)
LOGS_PATH = "./logs/"

# Path to the directory containing Tree-Ring watermarked images
TRAIN_PATH = '/home/dataset/outputs/watermarked'

def get_remover_img_batch(files_list,
                          secret_size,
                          batch_size=4,
                          size=(400, 400)):
    batch_cover = []
    batch_secret = []

    for i in range(batch_size):
        # Load Tree-Ring watermarked image
        img_cover_path = random.choice(files_list)
        try:
            img_cover = Image.open(img_cover_path).convert("RGB")
            img_cover = ImageOps.fit(img_cover, size)
            img_cover = np.array(img_cover, dtype=np.float32) / 255.
        except:
            img_cover = np.zeros((size[0], size[1], 3), dtype=np.float32)
        batch_cover.append(img_cover)

        # Generate a random secret message and cast to float32
        secret = np.random.binomial(1, .5, secret_size).astype(np.float32)
        batch_secret.append(secret)

    batch_cover = np.array(batch_cover, dtype=np.float32)
    batch_secret = np.array(batch_secret, dtype=np.float32)

    return batch_cover, batch_secret


def main():
    tf.reset_default_graph()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str)
    parser.add_argument('--num_steps', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--secret_size', type=int, default=100)  # Adjust based on your secret size
    parser.add_argument('--pretrained_encoder', type=str, default=None)
    parser.add_argument('--pretrained_decoder', type=str, default=None)
    args = parser.parse_args()

    EXP_NAME = args.exp_name

    # Collect image files
    image_extensions = ('*.jpg', '*.jpeg', '*.png')
    files_list = []
    for ext in image_extensions:
        files_list.extend(glob.glob(os.path.join(TRAIN_PATH, "**", ext), recursive=True))

    if not files_list:
        raise ValueError(f"No image files found in {TRAIN_PATH}")
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    height = 400
    width = 400
    secret_size = args.secret_size

    # Define placeholders
    image_pl = tf.placeholder(shape=[None, height, width, 3], dtype=tf.float32, name="input_image")  # Tree-Ring watermarked images
    secret_pl = tf.placeholder(shape=[None, secret_size], dtype=tf.float32, name="input_secret")
    remover_input_image_pl = tf.placeholder(shape=[None, height, width, 3], dtype=tf.float32, name="remover_input_image")

    # 세션 생성
    sess = tf.Session(config=config)

    # SavedModel 로드
    print("Loading encoder model from SavedModel...")
    from tensorflow.python.saved_model import loader
    loader.load(sess, [tag_constants.SERVING], ENCODER_MODEL_PATH, import_scope='encoder')
    graph = tf.get_default_graph()

    # saved_model.pb 파일 읽기
    from tensorflow.core.protobuf import saved_model_pb2
    saved_model = saved_model_pb2.SavedModel()
    with open(os.path.join(ENCODER_MODEL_PATH, 'saved_model.pb'), 'rb') as f:
        saved_model.ParseFromString(f.read())

    # MetaGraphDef 및 SignatureDef 가져오기
    meta_graph_def = saved_model.meta_graphs[0]
    signature_def = meta_graph_def.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    # 입력 및 출력 텐서 이름 가져오기
    input_secret_name = signature_def.inputs['secret'].name  # 'input_prep:0'
    input_image_name = signature_def.inputs['image'].name    # 'input_hide:0'
    output_residual_name = signature_def.outputs['residual'].name  # 실제 이름 사용

    # 텐서에 접근 (스코프 적용)
    input_secret = graph.get_tensor_by_name('encoder/' + input_secret_name)  # 'encoder/input_prep:0'
    input_image = graph.get_tensor_by_name('encoder/' + input_image_name)    # 'encoder/input_hide:0'
    output_residual = graph.get_tensor_by_name('encoder/' + output_residual_name)

    # 최종 출력 이미지 계산 및 그래디언트 차단
    output_img = input_image + output_residual
    output_img = tf.clip_by_value(output_img, 0, 1)
    output_img = tf.stop_gradient(output_img)  # 그래디언트 차단

    print("Loaded encoder model from SavedModel...!!!")

    # Remover 모델 정의
    with tf.variable_scope('stega_stamp_remover_model'):
        remover = StegaStampRemover(secret_size=secret_size)
        remover_output = remover([remover_input_image_pl, secret_pl])

    # 손실 함수 정의
    loss_op = tf.reduce_mean(tf.square(remover_output - image_pl))

    # 손실 값에 대한 스칼라 요약 추가
    loss_summary = tf.summary.scalar('loss', loss_op)

    # 이미지 요약을 위한 텐서 정의
    def image_summary(tag, images, max_outputs=3):
        return tf.summary.image(tag, images, max_outputs=max_outputs)
    
    # 원본 이미지, 인코더 출력 이미지, Remover 출력 이미지를 요약합니다.
    original_image_summary = image_summary('Original Images', image_pl)
    encoded_image_summary = image_summary('Encoded Images', remover_input_image_pl)
    remover_output_summary = image_summary('Remover Output Images', remover_output)
    
    # 모든 요약을 합칩니다.
    merged_summary = tf.summary.merge([loss_summary, original_image_summary, encoded_image_summary, remover_output_summary])

    print("Trainable variables:")
    for var in tf.trainable_variables():
        print(var.name)

    remover_vars = [var for var in tf.trainable_variables() if var.name.startswith('stega_stamp_remover_model/')]

    # 만약 remover_vars가 비어 있다면 오류 메시지를 출력
    if not remover_vars:
        raise ValueError("No trainable variables found for the Remover model.")    

    # 옵티마이저 정의
    global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.AdamOptimizer(args.lr)
    train_op = optimizer.minimize(loss_op, global_step=global_step_tensor, var_list=remover_vars)

    # Get optimizer variables
    optimizer_vars = [var for var in tf.global_variables() if 'Adam' in var.name or 'beta' in var.name]

    # Combine variables to initialize
    vars_to_initialize = remover_vars + optimizer_vars + [global_step_tensor]

    # Initialize variables
    init_op = tf.variables_initializer(vars_to_initialize)
    sess.run(init_op)

    # Saver for all variables
    saver = tf.train.Saver(var_list=remover_vars + [global_step_tensor], max_to_keep=5)

    writer = tf.summary.FileWriter(os.path.join(LOGS_PATH, EXP_NAME), sess.graph)

    total_steps = args.num_steps
    global_step = 0

    # tqdm으로 진행률 표시줄 생성
    with tqdm(total=total_steps, desc="Training Progress") as pbar:
        while global_step < total_steps:
            # Load a batch of images and random secret messages
            images, secrets = get_remover_img_batch(files_list=files_list,
                                                    secret_size=secret_size,
                                                    batch_size=args.batch_size,
                                                    size=(height, width))


            # 인코더 입력을 위한 feed_dict 생성
            feed_dict_encoder = {
                input_image: images,
                input_secret: secrets
            }

            # 세션을 실행하여 output_img 생성
            output_images = sess.run(output_img, feed_dict=feed_dict_encoder)

            # Remover 모델의 입력으로 사용
            feed_dict = {
                image_pl: images,                   # Tree-Ring watermarked images
                secret_pl: secrets,
                remover_input_image_pl: output_images
            }

            # Train the model and get the merged summaries
            _, loss_value, global_step_value, summary_str = sess.run(
                [train_op, loss_op, global_step_tensor, merged_summary],
                feed_dict=feed_dict)
            steps_completed = global_step_value - global_step
            global_step = global_step_value

            # tqdm 진행률 업데이트
            pbar.update(steps_completed)

            # 진행률 표시줄 설명 업데이트
            pbar.set_description(f"Step {global_step}, Loss: {loss_value:.4f}")

            # Log summaries to TensorBoard
            if global_step % 100 == 0:
                writer.add_summary(summary_str, global_step)

            # Save the model periodically
            if global_step % 10000 == 0:
                save_path = saver.save(sess, os.path.join(CHECKPOINTS_PATH, EXP_NAME, f"model_{global_step}.ckpt"))
                print(f"\nModel saved in path: {save_path}")

            # 훈련 중간에 이미지 저장
            if global_step % 1000 == 0:
                remover_outputs = sess.run(remover_output, feed_dict=feed_dict)
                for i in range(min(3, args.batch_size)):
                    original_img = (images[i] * 255).astype(np.uint8)
                    encoded_img = (output_images[i] * 255).astype(np.uint8)
                    remover_img = (remover_outputs[i] * 255).astype(np.uint8)
                    
                    Image.fromarray(original_img).save(os.path.join(LOGS_PATH, EXP_NAME, f"original_{global_step}_{i}.png"))
                    Image.fromarray(encoded_img).save(os.path.join(LOGS_PATH, EXP_NAME, f"encoded_{global_step}_{i}.png"))
                    Image.fromarray(remover_img).save(os.path.join(LOGS_PATH, EXP_NAME, f"remover_{global_step}_{i}.png"))

    writer.close()

if __name__ == "__main__":
    main()