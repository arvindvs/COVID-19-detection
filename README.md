# cs230
Run train script:
1. Enter environment
2. Run command 

python train.py -c <path to metadata csv> -d <path to folder containing images> -n <num of output classes> -e <num epochs of training> -s <directory to save artifacts>

Metadata CSV has rows with format
img_name, label
(All other columns ignored)