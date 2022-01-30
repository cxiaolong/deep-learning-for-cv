<h1 align='center'>CNN分类网络详解 </h1>

[TOC]

# LeNet-5

## 1 模型介绍

LeNet-5出自论文《Gradient-Based Learning Applied to Document Recognition》，是由$LeCun$ 于1998年提出的一种用于识别手写数字和机器印刷字符的卷积神经网络，其命名来源于作者$LeCun$的名字，5则是其研究成果的代号，在LeNet-5之前还有LeNet-4和LeNet-1鲜为人知。LeNet-5阐述了图像中像素特征之间的相关性能够由参数共享的卷积操作所提取，同时使用卷积、下采样（池化）和非线性映射这样的组合结构，是当前流行的大多数深度图像识别网络的基础。

## 2 模型结构
![](../libs/images/classification/LeNet/1.png)


​                                                             <center>   图1 LeNet-5网络结构图  </center>

LeNet-5虽然是早期提出的一个小网络，但是却包含了深度学习卷积神经网络的基本模块：`卷积层`、`池化层`和`全连接层`。如图1所示，LeNet-5一共包含7层（输入层不作为网络结构），分别由2个卷积层、2个池化层和3个连接层组成，网络的参数配置如表1所示，其中下采样层和全连接层的核尺寸分别代表采样范围和连接矩阵的尺寸。

<center> 表1 LeNet-5网络参数配置 </center>

| Layer Name | Kernel Size | Kernel Num | Stride | Padding |      Input Size      |     Output Size      |        Trainable params         |
| :--------: | :---------: | :--------: | :----: | :-----: | :------------------: | :------------------: | :-----------------------------: |
|   $C_1$    | $5\times5$  |    $6$     |  $1$   |   $0$   | $32\times32\times1$  | $28\times28\times6$  |  $(5\times5\times1+1)\times6$   |
|   $S_2$    | $2\times2$  |    $/$     |  $2$   |   $0$   | $28\times28\times6$  | $14\times14\times6$  |         $(1+1)\times6$          |
|   $C_3$    | $5\times5$  |    $16$    |  $1$   |   $0$   | $14\times14\times6$  | $10\times10\times16$ |             $1516$              |
|   $S_4$    | $2\times2$  |    $/$     |   2    |   $0$   | $10\times10\times16$ |  $5\times5\times16$  |         $(1+1)\times16$         |
|   $C_5$    | $5\times5$  |   $120$    |   1    |   $0$   |  $5\times5\times16$  | $1\times1\times120$  | $(5\times5\times16+1)\times120$ |
|   $F_6$    |     $/$     |    $/$     |  $/$   |   $/$   | $1\times1\times120$  |  $1\times1\times84$  |        $(120+1)\times84$        |
|  $Output$  |     $/$     |    $/$     |  $/$   |   $/$   |  $1\times1\times84$  |  $1\times1\times10$  |        $(84+1)\times10$         |

接下来，分别详解各层参数

**1、卷积层$C_1$**

> 手写数字数据集是灰度图像，输入为$32\times32\times1$的图像，卷积核大小为$5\times5$，卷积核数量为6，步长为1，零填充。最终得到的$C_1$的feature maps大小为（$32-5+1=28$）。可训练参数：$(5\times5+1)\times6$，其中有6个滤波器，每个滤波器$5\times5$个units参数和一个bias参数，总共需要学习156个参数，这些参数是权值共享的。

**2、下采样层$S_2$**

> 卷积层$C_1$之后接着就是池化运算，池化核大小为$2\times2$，LeNet-5池化运算的采样方式为**4个输入相加，乘以一个可训练参数，再加上一个可训练偏置，结果通过sigmoid**，所以下采样的参数个数是$(1+1)\times6$而不是零。

**3、卷积层$C_3$**

> 在LeNet-5中，$C_3$中的可训练参数并未直接连接$S_2$中所有的特征图（Feature Map），而是采用如图2所示的采样特征方式进行连接（稀疏连接）。具体地，$C_3$的前6个feature map（对应图2第一个红框的前6列）与$S_2$层相连的3个feature map相连接（图2第一个红框），后面6个feature map与$S_2$层相连的4个feature map相连接（图2第二个红框），后面3个feature map与$S_2$层部分不相连的4个feature map相连接，最后一个与$S_2$层的所有feature map相连。卷积核大小依然为$5\times5$，所以总共有$6\times(3\times5\times5+1)+6\times(4\times5\times5+1)+3\times(4\times5\times5+1)+1\times(6\times5\times5+1)=1516$个参数。在原论文中解释了使用这种采样方式原因包含两点：限制了连接数不至于过大（当年的计算能力比较弱）；强制限定不同特征图的组合可以使映射得到的特征图学习到不同的特征模式。

<div style="align: center"> 

![](../libs/images/classification/LeNet/2.jpg)


<div>


​                                                               <center> 图2 $S_2$与$C_3$之间的特征图稀疏连接</center>

**4、下采样层$S_4$**

> 与下采样层$S_2$类似，采用大小为$2\times2$，步距为2的池化核对输入feature maps下采样，输出feature maps大小为$5\times5$

**5、卷积层$C_5$**

> 与卷积层$C_3$不同，卷积层$C_5$的输入为$S_4$的全部feature maps，由于$S_4$层的16个图的大小为$5\times5$，与卷积核的大小相同，所以卷积后形成的图的大小为1x1。

**6、全连接层$F_6$**和$Output$

> $F_6$和$Output$层在图1中显示为全连接层，原论文中解释这里实际采用的是卷积操作，只是刚好在$5\times5$卷积后尺寸被压缩为$1\times1$，
> 输出结果看起来和全连接很相似。

## 3 模型特性
- 卷积网络使用一个3层的序列组合：卷积、下采样（池化）、非线性映射（LeNet-5最重要的特性，奠定了目前深层卷积网络的基础）
- 使用卷积提取空间特征
- 使用映射的空间均值进行下采样
- 使用$tanh$或$sigmoid$进行非线性映射
- 多层神经网络（MLP）作为最终的分类器
- 层间的稀疏连接矩阵以避免巨大的计算开销



# AlexNet
