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

The Conda install yaml doesn't currently work due to some import issue, so a venv is used for now.

Run the attack:

```bash
# rinse.py watermarked_path output_path strength n_rinse [--model StableDiffusionMODEL]
# Model can be "CompVis/stable-diffusion-v1-4" or (by default) "stabilityai/stable-diffusion-2-1" or something else
# Output folder will be created if it doesn't already exist.
python rinse.py ./watermarked_example/ ./attacked_example 10 1
```