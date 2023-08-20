[中文](https://github.com/Phoenix8215/FCN_KITTI/blob/master/README-zh.md)
## Using FCN for road segmentation (KITTI dataset)
![](https://img.shields.io/badge/segmentation-FCN--KITTI-red)

### Install

```shell
git clone https://github.com/Phoenix8215/FCN_KITTI  # clone
cd 
pip install -r requirements.txt  # install
```

### Instructure manual

- use`make_tsv.py`to generate the train set,validation set and test set with `tsv` format，the set of validation and testing is 80%,20% respectively.

- backbone use VGG16，loss function is binary cross entropy，training 60 epochs.
- use`python3 train.py`to train the model，and visual the process of training.

<img src="https://pic.imgdb.cn/item/6311ff2116f2c2beb1217eee.png" style="zoom:50%;" >
<img src="https://files.downk.cc/item/64e174c7661c6c8e549ef560.jpg" alt="blob">
<img src="https://pic.imgdb.cn/item/6311ff2116f2c2beb1217f04.png" style="zoom:50%;" >
<img src="https://pic.imgdb.cn/item/6311ff2216f2c2beb1217f26.png" style="zoom:50%;" >
<img src="https://pic.imgdb.cn/item/6311ff2216f2c2beb1217f2f.png" style="zoom:50%;" >

- use `python3 test.py`to predict ,outputs are stored in`output`文folder.

![](https://pic.imgdb.cn/item/6311fed316f2c2beb1210144.png)

- generated weight files which is in`ckpt`folder.

