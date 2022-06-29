# IFUNet
RIFE with IFUNet, FusionNet and RefineNet

* Download the pretrained models from [here](https://drive.google.com/file/d/1psrM4PkPhuM2iCwwVngT0NCtx6xyiqXa/view?usp=sharing).
* Unzip and move the pretrained parameters to train_log/\*

**Run Video Frame Interpolation**
```
python3 inference_video.py --img=imgs/ --scale=1.0 --multi=2
```
