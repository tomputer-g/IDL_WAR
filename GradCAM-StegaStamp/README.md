# GradCAM on StegaStamp

### Overview

This is an application of GradCAM on the StegaStamp image decoder in order to extract the locations of interest for the decoder model. Our hypothesis is that these locations are where the decoder model seeks secret bits from, and therefore contains the residuals for StegaStamp. We can then apply localized attacks to these areas of interest to avoid unnecessarily damaging the input image.

### Setup

Set up conda environment:
```
conda env create -f environment.yml
conda activate StegaStamp
```

### Reproduce Results


### Generate Attacked Images

Note: there are around 15 GB of result images for GradCAM, residual, residuals different message, and randomized attacks. To produce fewer images, use the `attack_*.py` file corresponding to the attack you want to run directly rather than `generate_attacked.sh`.

#### Generate GradCAM Maps
```
python save_gradcam.py /path/to/StegaStampModel --wm_images_dir /path/to/watermarked_images --output_dir /path/to/output_dir
```

#### Generate GradCAM Attacked Images
Uses GradCAM map to attack the pixels most relevant to the StegaStamp decoder.
```
bash generate_attacked.sh gradcam /path/to/StegaStampModel /path/to/watermarked_images /path/to/generated_gradcams
```

#### Generate Blur Attack Images
Generic blurring attack.
```
bash generate_attacked.sh blur /path/to/watermarked_images
```

#### Generate Residual Attack Images
Uses the residuals from StegaStamp instead of GradCAM to attack the images.
```
bash generate_attacked.sh residuals /path/to/watermarked_images /path/to/unwatermarked_images
```

#### Generate Residual Attack Images
Uses the residuals from StegaStamp instead of GradCAM to attack the images, but uses the residuals from a different watermark message. Imitates what happens if you use a residual attack and don't know the original watermark message.
```
bash generate_attacked.sh residuals_diff_message /path/to/watermarked_images /path/to/watermarked_different_message_images /path/to/unwatermarked_images
```

### Visualizations

```bash
python decode_gradcam.py <model_checkpoint> --image <watermarked_image>
```

This should produce a figure where the image is side by side with the GradCAM activations.

### Notes

Get the model checkpoint from Dongjun, and the watermarked image also from Dongjun. The unwatermarked corresponding images should be from our google drive.

See StegaStamp decoder model definition [here](https://github.com/tancik/StegaStamp/blob/master/models.py#L82-L93). Note that at the end of the same file, the outputs of the decoder model is passed through a Sigmoid layer and then a Round operation (which destroys the gradient information), so we use the Sigmoid layer instead for the 'output'.

The original StegaStamp code is in Tensorflow v1, so GradCAM was also implemented in TF V1. 

### Citations

The `decode_gradcam.py` code is adapted from `decode_image.py` from the StegaStamp authors' original repository here (https://github.com/tancik/StegaStamp/tree/master). A large amount of code is in reference to this.

