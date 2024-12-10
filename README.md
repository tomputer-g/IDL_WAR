  # IDL-WAR: Invisible Watermarking And Robustness

## Overview

This is the Final Project for Team 3 of Carnegie Mellon University's Introduction to Deep Learning (11-785) course in the fall semester of 2024. Our team members are Dongjun Hwang, Sungwon Woo, Tom Gao, and Raymond Luo (all equal contribution).

Our project concerns Invisible Watermarks - messages that are encoded into AI-generated images in order to detect and identify sources of AI generation. We primarily investigate StegaStamp and Tree-Ring as watermarking methods. In a nutshell, StegaStamp is an **image-space** stenographic algorithm that hides the given message into the pixels of the image so that they are resistant against distortions of images; Tree-Ring embeds the message in the Fourier-Transformed initial **latent-space** noise vector, which can be retrieved through the reverse-DDIM process. We explore how these two watermarks can be combined by placing StegaStamp on top of Tree-Ring naively and with a remover architecture. Furthermore, we explore how attacks can reduce their image quality degradation by only attacking specific important pixels. We use a base blurring attack and use GradCAM to localize which pixels are significant.

A copy of our final report will be linked when it is made publicly available. You can view our quick 5-minute [final presentation video here](https://youtu.be/0vwFG1HSrUE).

## Abstract

As Generative AI continues to become more accessible, the case for robust detection of generated images in order to combat misinformation is stronger than ever. Invisible watermarking methods act as identifiers of generated content, embedding image- and latent-space messages that are robust to many forms of perturbations. The majority of current research investigates full-image attacks against images with a single watermarking method applied. We introduce novel improvements to watermarking robustness as well as minimizing degradation on image quality during attack. Firstly, we examine the application of both image-space and latent-space watermarking methods on a single image, where we propose a custom watermark remover network which preserves one of the watermarking modalities while completely removing the other during decoding. Then, we investigate localized blurring attacks (LBA) on watermarked images based on the GradCAM heatmap acquired from the watermark decoder in order to reduce the amount of degradation to the target image. Our evaluation suggests that 1) implementing the watermark remover model to preserve one of the watermark modalities when decoding the other modality slightly improves on the baseline performance, and that 2) LBA degrades the image significantly less compared to uniform blurring of the entire image. 

## Repository Information

Each folder at the root of this repository is a self-contained piece of the project that is installed and run individually. Each folder contains a README which contains all the necessary details for installing the relevant part, and sample commands for running them.

* GradCAM-StegaStamp: Localized Blurring Attack (LBA) implementation, where we use the StegaStamp decoder network with GradCAM to localize areas of denser information and run blurring kernels on them.

* Remover-StegaStamp: Implementation for the double watermarking pipeline, including encoding, decoding, and detection of the watermarks, with the pretrained weights for our watermark remover network.

* latent_rinsing_attack: Implementation of the Rinse-nxDiff and Regen-nxDiff attacks created by Zhao et al. (WatermarkAttacker) and An et al. (WAVES Benchmark). Regeneration attacks attack the image by getting its latent, adding noise to the latent space of the image, then regenerating the image from the noised latent. Rinsing attacks do this `n` times in a row.

* tree-ring: Implementation of the Tree-Ring watermarking algorithm. Scripts are included to apply, evaluate, adn get reversed latents for images.

* warm-up-kit submodule: We found that the NeurIPS 2024 Erasing the Invisible Competition had a suite of watermark performance and image quality metrics that are helpful in evaluating the results of watermarking attacks. We included this repository as a submodule in our project.
