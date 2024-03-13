Deep Person Re-Identification for Motorcycle Racing
===========
This repository is a modified version of Torchreid,
updated to support Performance Photo for person re-identification, along with a few other neat tricks.

For information on getting started with this codebase, we recommend you first get familiar with its parent repository: 
https://github.com/KaiyangZhou/deep-person-reid

The muddy racer re-identification (MUDD) dataset can be downloaded here: https://drive.google.com/file/d/1s9AImKlUErj4ZDOrBYxKEs_iar4BUnbp/view?usp=drive_link 

For more information on the structure, see the dataloader located at `torchreid/data/datasets/image/performance.py`. Furthermore, the configureation files located in `configs/performancephoto` outline the proper way to invoke `scripts/PerPh_main.py` for use with MUDD.

For more information on Contrastive Multiple Instance Learning (CMIL), see `configs/market` or `configs/performancephoto`. Any configuration file with `mil` in the name launches CMIL. 
