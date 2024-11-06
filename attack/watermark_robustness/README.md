

## Diffusion Purification

In this part, the procedure to apply watermarking methods to images, and evaluting them against diffusion purification attack is explained. Evaluation against **common perturbation** and **image editing** is done in a similar way.


## Preparing Seed Data
Specify path to competition dataset

## Preparing Pretrained Diffusion model
Run _bash_download_models.sh

## Evaluation Against Attacks

1. Run _bash_eval_wm.sh. 

2. Choose whichever --wm_method , just stop the code before evaluating metric. 
3. Edit output image directory at evaluate_watermark.py





## Citation

```bibtex
@inproceedings{saberi2023robustness,
    title={Robustness of AI-Image Detectors: Fundamental Limits and Practical Attacks},
    author={Saberi, Mehrdad and Sadasivan, Vinu Sankar and Rezaei, Keivan and Kumar, Aounon and Chegini, Atoosa and Wang, Wenxiao and Feizi, Soheil},
    booktitle = {ICLR},
    year = {2024}
}
```