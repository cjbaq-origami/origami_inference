# origami_inference
Private inference using hardware enclaves
## Folder Structure
Folder|Description|
---|---
slalom| Contains code for baselines and Origami
cGAN-reconstruct-image/split_layer6 |  Contains code for collecting intermediate feature maps used for training c-GAN
cGAN-reconstruct-image/code_split_layer6 | contains code for training c-GAN networks

### To collect intermediate feature maps
```
cd cGAN-reconstruct-image/split_layer6
python36 split_vgg16.py -p "path to the image dataset"
```

### To train the c-GAN networks on intermediate feature maps
```
cd cGAN-reconstruct-image/code_split_layer6
python36 main.py --gpu 0 -i "path to intermediate feature maps"
```
