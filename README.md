# Colorizer

Image colorization with multiple approaches, with PyQt5 GUI.

我们制作了共计三种上色、重上色的算法，并使用pyQt5进行了页面的工程整合。

总共实现的内容如下：

一、传统方法上色

1、Colorization_By_Reference

```
 论文复现《Transferring Color to Greyscale Images》 
 Article  in  ACM Transactions on Graphics · July 2002
```

​		1)  灰度图片上色【全局参考上色与交互式指定区域上色】

​		2）彩色图重上色【指定区域重上色】

2、Colorization_By_Optimization

```
论文复现《Colorization using Optimization》 Siggraph 2004
```

​		1)  灰度图片上色【指定图片上色或交互式画笔上色】

​		2）彩色图重上色（优化方法）

​				【指定图片重上色或交互式画笔上色，交互式指定区域上色】

​		3）灰度视频优化上色【参考帧素材与视频素材】

二、深度学习方法上色

​		xxxx 待补充

同时，我们能够实现，调用摄像头，实时获取一帧进行部分区域的重上色。

以下为文件目录结构的说明：

Colorizer.py为全局入口文件，运行此py文件即可运行工程。

​		第一次运行可能会下载权重数据集，时间较久，请耐心等待，也可以到百度网盘中下载相应数据，复制入其正在下载的路径中即可。

链接：https://pan.baidu.com/s/1v9CJeP29S0GBCKJDsnAoZQ 
提取码：u71i 

1、colorization_by_custom_method	传统方法上色所在文件夹

​	1）colorization_by_optimization 优化方法对图片上色

​	2）colorization_by_optimization_video_part 优化方法对视频上色

【由于全局优化计算时间较慢，故此部分未整合进入工程界面中，但是提供了示例结果以及示例内容，限于算力，仅对50*50的30帧视频，使用3帧参考站来进行全局优化，效果不是很好】

​	3）colorization_by_reference 参考方法对图片上色

2、colorization_by_learning 深度学习方法上色所在文件夹

3、images 中为一些测试图片素材，可供使用