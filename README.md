# MCSC（Multi-Scale Cross Contrastive Learning for Semi-Supervised Medical Image Segmentation）
bmvc 2023 paper: Multi-Scale Cross Contrastive Learning for Semi-Supervised Medical Image Segmentation


## Requirements
Some important required packages include:
* [Pytorch][torch_link] version >=1.7.1
* TensorBoardX
* Python == 3.8 
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......

Follow official guidance to install [Pytorch][torch_link].

[torch_link]:https://pytorch.org/

# Usage

1. Clone the repo:
```
git clone https://github.com/kathyliu579/MCSC.git
cd MCSC
```
2. Download the processed data and put the data in `../data/ACDC`, please read and follow the [README](https://github.com/Luoxd1996/SSL4MIS/tree/master/data/).

3. Train the model
```
cd code
python python train_MCSC_2D_256_28.py or bash/sh train_acdc_unet_semi_seg.sh
```

4. Test the model
```
python test_2D_fully.py or bash/sh test_acdc_unet_semi_seg.sh
```

## References
* [SwinUnet](https://github.com/HuCaoFighting/Swin-Unet)
* [SSL4MIS](https://github.com/HiLab-git/SSL4MIS) 

# Citation
@misc{liu2023multiscale,
      title={Multi-Scale Cross Contrastive Learning for Semi-Supervised Medical Image Segmentation}, 
      author={Qianying Liu and Xiao Gu and Paul Henderson and Fani Deligianni},
      year={2023},
      eprint={2306.14293},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
