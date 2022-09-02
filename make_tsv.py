import glob
import copy
test_list = sorted(glob.glob("./data_road/testing/image_2/*"))
train_list = sorted(glob.glob("./data_road/training/image_2/*"))
gt_of_train_list = sorted(glob.glob("./data_road/training/gt_image_2/*"))


file = open("full.tsv","w")
for item in gt_of_train_list:
    temp_list = item.split("/")[2:]
    temp_list2 = temp_list.copy()
    temp_list3 = temp_list[-1].split("_")
    target = temp_list3[0] + "_" + temp_list3[-1]
    temp_list2[-1] = target
    temp_list2[-2] = "image_2"
    ret1 = "/".join(temp_list)
    ret2 = "/".join(temp_list2)
    ret = ret2 + "\t" + ret1 + "\n"
    file.writelines([ret])

file.close()

import random
from random import randint

oldf = open('full.tsv', 'r')
trainf = open('train.tsv', 'w')
valf = open("val.tsv", "w")
n = 0
# sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素
train_list = random.sample(range(0, len(train_list)), int(0.8*len(train_list)))
val_list = list(set(range(0, len(train_list))) - set(train_list))

lines = oldf.readlines()
for i in train_list:
    trainf.write(lines[i])

for j in val_list:
    valf.write(lines[j])



oldf.close()
trainf.close()
valf.close()







