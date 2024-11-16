# Environment setup
Set up conda environment:

```
conda env create -f environment.yml
conda activate tree-ring
```

Download data:

```
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
```

# Generate Tree-Ring Images
```python apply_tree_ring.py```

Inputs:
```
Options:
  --hf_dataset TEXT               Dataset you are working with.  [default:
                                  phiyodr/coco2017]
  --split TEXT                    Split to use  [default: validation]
  --output_folder TEXT            Folder to put results in.  [default:
                                  outputs]
  --channel INTEGER               Channel to put tree-ring watermark in.
                                  [default: 0]
  --num_files_to_process INTEGER  The number of files to actually process.
                                  [default: -1]
  --resume                        Resume from previous run.  [default: True]
  --help                          Show this message and exit.
```

Outputs:
```
outputs/
├── captions
|   └── text files with the caption used to generate each image
├── keys
|   ├── .pt files containing the keys used to watermark the images
|   └── can be loaded using torch.load()
├── masks
|   ├── .pt files containing the masks used to watermark the images
|   └── can be loaded using torch.load()
├── unwatermarked
|   └── unwatermarked images
├── watermarked
|   └── watermarked images
└── processed.txt
```

# Evaluate Tree-Ring

Without attacks:
```python eval_tree_ring.py```

# References
## Tree-Ring
```bibtex
@misc{wen2023treeringwatermarksfingerprintsdiffusion,
      title={Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust}, 
      author={Yuxin Wen and John Kirchenbauer and Jonas Geiping and Tom Goldstein},
      year={2023},
      eprint={2305.20030},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2305.20030}, 
}
```