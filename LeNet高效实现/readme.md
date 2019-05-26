## 更高效的实现LeNet,并添加了绘制loss虽训练轮数变化的图和准确率随训练轮数变化的图
## 提升效率的地方在卷积的实现和池化的实现，调用了numpy的一些函数，有些复杂，需要仔细研究函数的使用方法，大家可以对比着看


#  LeNet卷积神经网络Python底层实现代码
##  python版本是3.7，需要安装numpy库
##  在运行前需要对dataset_loader函数中，加载数据的路径进行更改，替换为你下载的MNIST数据所存放的路径。 
##  LeNet.ipynb可以直接用jupyter notebook打开
##  LeNet.py格式的代码可以在命令行进行运行 python LeNet.py或者 python3 LeNet.py
