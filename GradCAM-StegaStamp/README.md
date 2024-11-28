# GradCAM on StegaStamp

### Overview

This is an application of GradCAM on the StegaStamp image decoder in order to extract the locations of interest for the decoder model. Our hypothesis is that these locations are where the decoder model seeks secret bits from, and therefore contains the residuals for StegaStamp. We can then apply localized attacks to these areas of interest to avoid unnecessarily damaging the input image.

### Setup

Set up a Conda environment using the given `environment.yml` and activate it. Then, run:

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

