## 使用FCN模型对KITTI数据集进行道路分割

![](https://img.shields.io/badge/segmentation-FCN--KITTI-red)

### 安装

```shell
git clone https://github.com/ultralytics/yolov5  # clone
cd 
pip install -r requirements.txt  # install
```

### 使用手册

- 使用`make_tsv.py`生成训练集，验证集和测试集的`tsv`格式的文件，其中验证集和测试集的比例分别为80%和20%

- 骨干网络采用VGG16，损失函数使用二元交叉熵，训练60个轮次
- 执行训练操作`python3 train.py`，并将训练过程可视化出来

<img src="https://pic.imgdb.cn/item/6311ff2116f2c2beb1217eee.png" style="zoom:50%;" >
<img src="https://pic.imgdb.cn/item/6311ff2116f2c2beb1217efa.png" style="zoom:50%;" >
<img src="https://pic.imgdb.cn/item/6311ff2116f2c2beb1217f04.png" style="zoom:50%;" >
<img src="https://pic.imgdb.cn/item/6311ff2216f2c2beb1217f26.png" style="zoom:50%;" >
<img src="https://pic.imgdb.cn/item/6311ff2216f2c2beb1217f2f.png" style="zoom:50%;" >

- 对测试集进行预测`python3 test.py`,生成的预测结果存放在`output`文件夹中

![](https://pic.imgdb.cn/item/6311fed316f2c2beb1210144.png)

- 生成的权重文件存放在`ckpt`中

