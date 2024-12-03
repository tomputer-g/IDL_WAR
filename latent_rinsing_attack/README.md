# What?

This is an adaptation of the WAVES Regeneration attack inspired by Rinse-4xdiff. We also cite the WatermarkAttacker repository, from which ReSDPipeline class is used. 

### Parameters

* Model: some version of pre-trained stable diffusion for now. The paper original code used Stable Diffusion V1.4 and we are also using V2.1.

* Strength: Larger = higher distortion which degrades quality further.

* n: How many times the same model attack is performed. This is just how many times the model runs successively.

### Running the attack

Start by creating an environment. Python3.11 was used:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Alternatively, create a Conda environment:

```bash
conda env create -f environment.yml

```

Run the attack:

```bash
# rinse.py watermarked_path output_path strength n_rinse [--model StableDiffusionMODEL]
# Model can be "CompVis/stable-diffusion-v1-4" or (by default) "stabilityai/stable-diffusion-2-1" or something else
# Output folder will be created if it doesn't already exist.
python rinse.py ./watermarked_example/ ./attacked_example 10 1
```

### Getting Metrics

The results in our report are generated using the NeurIPS 2024 Erasing the Invisible Competition warm-up kit, which has a variety of metrics on watermark performance and image quality. For more information, please see https://github.com/erasinginvisible/warm-up-kit/tree/main.

The evaluation requires three sets of images: The original images (unwatermarked), watermarked images, and attacked watermarked images. By default, the kit evaluates the performance and quality of the 5000 MSCOCO images (which is also provided on their github page), and expects images numbered from 0.png to 4999.png for each of the three sets of images. 

For convenience we added a submodule for this repository at the root of our repository. Simply clone it and follow their setup steps. Here's an example command and its output:

```bash
erasinginvisible eval --path <Attacked> --w_path <Unattacked Watermarked> --uw_path <Unwatermarked>


####################

# Evaluation results:
### Watermark Performance:
Accuracy: 54.64%
AUC Score: 56.15%
TPR@0.1%FPR: 0.02%
TPR@1%FPR Score: 0.44%

### Image Quality:
Legacy FID: 3.650597e+00 +/- 0.000000e+00
CLIP FID: 4.318322e-01 +/- 0.000000e+00
PSNR: 2.375581e+01 +/- 2.497472e+00
SSIM: 7.339683e-01 +/- 9.565198e-02
Normed Mutual-Info: 1.212993e+00 +/- 5.302735e-02
LPIPS: 1.027168e-01 +/- 3.272342e-02
Delta Aesthetics: 1.641676e-01 +/- 3.112150e-01
Delta Artifacts: -5.591750e-02 +/- 1.820317e-01

Warmup kit evaluation completed.
####################
```


### Citations

Credit to the authors of the paper, WAVES: Benchmarking the Robustness of Image Watermarks, where we adapted the latent space rinsing code:

https://github.com/umd-huang-lab/WAVES/blob/main/regeneration/regen.py

Credit to Zhao et al., authors of the paper, Invisible Image Watermarks Are Provably Removable Using Generative AI, where we reused their ReSDPipeline code to perform the attack.

https://github.com/XuandongZhao/WatermarkAttacker/blob/main/regen_pipe.py

