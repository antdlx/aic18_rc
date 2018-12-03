## Model-2：capsule-mrc
这个Model主要借鉴了freefuiiismyname大神开源的代码，下面的介绍也大多是源自他本人的项目，我仅仅添加了一点我的理解和修改。
这个模型是基于BiDAF那个框架进行的修改，主要的变化是把BiDAF框架中的Attention Flow Layer修改成了大赛baseline中用的到的multiway Attention，这个multiway attention可以去搜一下这篇论文《Multiway Attention Networks for Modeling Sentence Pairs》，然后modeling Layer去掉，取而代之的是一个capsule network,最后把编码过的alternatives与胶囊网络的输出点乘下。

如果希望了解capsule network，我推荐下面这两篇文章：

[CapsNet ——胶囊网络原理](https://blog.csdn.net/godwriter/article/details/79216404)

[先读懂CapsNet架构然后用TensorFlow实现，这应该是最详细的教程了](https://zhuanlan.zhihu.com/p/30753326)

这个Model包括了2个子模型用来做ensemble，分别是ver81和ver84两版：
* ver81：用25w数据做的word2vec，lr=0.0005，acc=73.85
* ver84: 用25w数据做的word2vec，使用了cosin_restart的learning rate变换方式,acc=73.74

## 模型图
![pic1](https://github.com/antdlx/aic18_rc/blob/master/capsuleNet/model.png)

## 模型思路
**步骤1、问题编码**

使用bi-LSTM对query进行编码，它将作为passage和候选答案的背景信息。（无论是passage还是候选答案，都是基于query出现的，故而query应当作为两者的上下文）

**步骤2、形成候选答案的各自意义**

使用bi-LSTM对三个候选答案进行编码，以步骤1输出的state作为lstm的初始状态，使它们拥有问题的上下文信息。编码后将每个候选答案看作capsule，分别代表了三个不同事件。

**步骤3、形成passage对问题的理解**

对passage进行LSTM（使passage每个词语拥有上下文意思，state初始化同上）、match(与问题信息交互)、fuse（信息融合）、cnn（抽取关键信息）之后，形成N个特征capsule，代表了passage根据问题而抽取出的信息。

**步骤4、以候选答案为中心，对passage信息进行聚类**

将passage中抽取出的信息，转换为候选答案capsule。当某答案编码与passage信息相近时，信息更容易为它提供支撑；反之，它受到的支撑将减少。经过几轮的动态路由迭代过程后，最终capsule的模长代表了该答案存在的程度。Softmax后，求出每个候选答案作为答案的概率。

## Usage
download the [word2vec](https://pan.baidu.com/s/1Izg778MiUlcoqNMimWKjNQ) and put it into 'data/w2v'

直接运行run**.py即可，需要注意的是你需要指定两个参数才可以：
* --mode：test/dev/train/prepro  需要首先选择prepro模式，预处理好数据，然后可以根据需要去选择test/dev/train 模式
* --input: 一个路径，test/dev/train模式需要输入相应的文件路径，prepro模式需要输入test的文件路径
