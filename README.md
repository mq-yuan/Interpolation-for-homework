![](https://typora-ilgzh.oss-cn-beijing.aliyuncs.com/202303241424868.png)

# 图像插值

## 图像插值理论

插值指的是利用已知数据去预测未知数据，图像插值则是给定一个像素点，根据它周围像素点的信息来对该像素点的值进行预测，如何给这些值赋值，就是图像插值所要解决的问题。

本次实验主要关注：最近邻插值(Nearest neighbor interpolation)、双线性插值(Bilinear interpolation)和径向基函数插值(RBF interpolation)，其中径向基函数插值选用TPS(thin plate spline)和Gaussian两种径向基函数。

## 环境依赖

- `pip install -r requirements.txt`.

## 测试

在windows系统且确保已安装python3的情况下，点击`run_me.bat`文件可直接运行。

## 文件架构

测试用图片中已附带`Image`文件夹如下：

```txt
Image
├─person # 个人图片
|   ├─person_01.bmp
|   ├─person_02.bmp
|   └person_03.bmp
├─mask # 遮挡用图片
|  ├─butterfly_words_1.bmp # 文字遮挡
|  ├─butterfly_words_2.bmp # 文字遮挡
|  ├─Random scribbles_1.bmp # 随机涂鸦
|  └Random scribbles_2.bmp # 随机涂鸦
├─example # 图像处理常用图片
|    ├─example_01.tiff
|    ├─example_02.bmp
|    └example_03.bmp
```

一共有两种测试方法：1. 根据mask遮挡进行插值 2. 随机丢弃一定比例的像素点后进行插值。

- 根据mask遮挡插值：先在`Image/person`或`Image/example`中选择测试图片，后在`Image/mask`中选择遮挡图片，最后选择mask模式。
- 随即丢弃插值：在`Image/person`或`Image/example`中选择测试图片，输入随机丢弃像素点比例，最后选择随机丢弃模式。

结果会保存在`Ans`文件夹下，文件夹结构如下：

```txt
Ans
├─person_02 # 人测试图片文件夹2
|     ├─Randomly remove 20% # 随即丢弃20%文件夹
|     |          ├─Ans.png # 结果比较图片
|     |          ├─BNI.bmp # BNI插值图片
|     |          ├─Masked.bmp # 遮挡或丢弃后图片
|     |          ├─NNI.bmp # NNI插值图片
|     |          ├─Orignal.bmp #  原始图片
|     |          ├─RBF_GAUSSIAN.bmp # RBF_GAUSSIAN插值图片
|     |          └RBF_TPS.bmp # RBF_TPS插值图片
|     ├─Randomly remove 10% # 随机丢弃10%文件夹
|     |          ├─....... # 同上
|     ├─Random scribbles_2 # 随机涂鸦2文件夹
|     |          ├─....... # 同上
|     ├─Random scribbles_1 # 随机涂鸦1文件夹
|     |          ├─....... # 同上
|     ├─butterfly_words_2 # 文字遮挡2文件夹
|     |          ├─....... # 同上
|     ├─butterfly_words_1 # 文字遮挡1文件夹
|     |          └....... # 同上
├─person_01 # 个人测试图片文件夹1
|     ├─...... # 同上
├─example_03
|     ├─......
├─example_02
|     ├─......
```

**注意**：测试图片尽量在512*512以下，否则计算时间会过长。

