## Model-1 QANet
This model is based on QANet, but it has 2 submodel. In this folder, you can see 3 version of model, ver20 belongs to submodel-1, ver60 and ver646 belong to submode-2. I use these 3 version to ensemble the model. The details of these 3 version are:

Version | Kernel Size Char | Kernel Size Conv | Hidden Size
---|---|---|---
ver20| 2 | 7 | 96
ver60| 2 | 7 |96
ver646| 1| 4 | 64

Kernel Size Char: the kernel size used in char embedding layer.

kernel Size Conv : the kernel size used in model encoder layer

all of them use params below:
* w2v ： jwe_size300.txt，this is a word embedding trained by spliting Chinese characters into components, for example, we split “好” into "女" and "子". My partner Zhang does this excellent job. You can download it from [HERE](https://pan.baidu.com/s/1eKa7F-OBGQgLSsOaTtJDxg), the password is "qt16".
* context length: 100, the max length for context, i want to keep context words num below 100.
* query length: 30, the max length for query, i want to keep query words num below 30.

If you want to know my code clearly, you should learn QANet principle first.
And I have written a blog for you :D

[彻底弄懂QANet](https://antdlx.com/qanet/)

## SubModel-1
![submodel1](http://cdn.antdlx.com/qa20.png)
1. I add an alternatives embedding layer
2. I change model encoder layer's encoder block num from 3 to 2
3. I change output layer

## SubModel-2
![submodel2](http://cdn.antdlx.com/qa60.png)
1. I change model encoder layer's encoder block num from 3 to 2
2. I change output layer

## Usage
run config_xx_vxx.py. You need to set at least 2 params:
* --mode：test/valid/train/debug
* --input: test/valid/train file path, especially , debug mode uses train path

Be careful, the results are in answers folder, but these are tmp file, you can run "vote_ser_new_word.py" to get the final answers or edit codes by yourself. And thanks for my partner Liu wrote this .py
