# Demo repository

### Datasets
Download the two datasets ADEPT and BOUNCINGBALLS from https://nc.mlcloud.uni-tuebingen.de/index.php/s/nfoe5Q9RPXx4Btd
and place the respective folders into data/data/video.

### Models
Pretrained Loci-Looped models for both datasets can be found in out/pretrained1.

### Evaluation
Run the eval.sh script to evaluate the pretrained models with 'sh eval.sh'. (Comment out one of the lines in eval.sh) 
OR copy and paste the command line and run it in the terminal. 

### Modifications 
At the very beginning of the scripts: scripts/evaluation_adept.py and scripts/evaluation_bb.py
a section SETTINGS is included. Here you can control some of the model parameters and run evaluations with different settings.

Note that the evaluation of the Bouncingballs dataset is relatively fast whereas the ADEPT evaluation is much slower.