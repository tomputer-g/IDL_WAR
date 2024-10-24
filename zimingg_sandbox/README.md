# What?

This is an adaptation of the WAVES Regeneration attack inspired by Rinse-4xdiff. We also cite the WatermarkAttacker repository, from which ReSDPipeline class is used. 

regen.ipynb currently performs attacks on a single input, watermarked image. To use this on Colab (or anywhere else) simply upload the `regen.ipynb` file and run it from top to bottom, installing any required dependencies.

### Parameters

* Model: some version of pre-trained stable diffusion for now. The paper original code used Stable Diffusion V1.4 and we are also using V2.1.

* Strength: Larger = higher distortion which degrades quality further.

* n: How many times the same model attack is performed. This is just how many times the model runs successively.

### Evaluation

Eventually we will use metrics and be able to generate this programmatically. One current method is to run attacks on the included 'watermarked.png' and submit results at https://huggingface.co/spaces/furonghuang-lab/Erasing-Invisible-Demo, which returns Q/P and some other scoring conditions as well.