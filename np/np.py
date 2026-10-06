import numpy as np
a=np.ones((2,3,4))
b=np.array([1,2,3])
print(b)
print(b.shape)
"""
array.shape从外向内表示维度，例如b就有3个元素，每个元素0维
如a第一个维度是2，有两个元素，第二个维度3，这两个元素每个包含三个元素；
第三个维度4，这三个元素的元素每个包含4个元素。因此a.shape=(2,3,4)
索引也是从外到内嵌套表示
"""
print('----------np数组的基本概念-----------')
c=np.array([[[1,2],[1,2]],[[1,2],[1,2]]])
print(c)
print(c.shape)
print(c[0,0])
print(c[1,1])
print("--------------------")
print("Shape")
print(a.shape)
print("Sum over axis 0")
print(np.sum(a,axis=0))
"""
a是一个三维嵌套列表，有2个3*4的全1矩阵
对第一个维度压缩求和，就是把两个3*4矩阵变成一个3*4全2矩阵
np.sum(array,axis=number)
就是对一个高维数组的某个维度压缩求和，叠合切片，其他维度保持不变
"""
print('-----------索引访问和切片--------------')
a=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
b=a[:2,1:3]
print("array slice b")
print(b)
print("slice shape")
print(b.shape)
"""
索引操作是按照维度逐个先后操作，逐步地缩小范围

修改切片，也会修改整个数组

这里，第一个切片操作:2，表示操作第一个维度，取a[0],a[1]
第二个切片操作[1:3]表示操作第二个维度，取到a[0][1:3],a[1][1:3]
"""
print("try modifying the slice b, and a is modified too.")
print(a[0])
b[0][0]=5
print(a[0])
print('------------整数索引-----------------')
a=np.array([[1,2],[3,4],[5,6]])
print(a[[0,1,2],[0,1,0]])
#下列是等价的显式写法
print(np.array([a[0,0],a[1,1],a[2,0]]))
print(a[[0,0],[1,1]])
#等价于
print(np.array([a[0,1],a[0,1]]))
print("----------数据类型--------------------")
"""
创建np数组可以手动指定第二个参数dtype
int64、float64
"""
"""
算术运算是element-wise，即逐元素进行
"""
print("------------算术运算------------------")
x=np.array([[1,2],[3,4]],dtype=np.float64)
#x是一个np数组，它的数据类型手动指定为float64
y=np.array([[5,6],[7,8]],dtype=np.float64)
#同上
print(x+y)
#每个元素相加
print(np.add(x,y))
#等价于上面一行

print(x-y)
#逐个相减
print(np.subtract(x,y))
#等价于上面

print(x*y)
#逐个相乘
print(np.multiply(x,y))

print(x/y)
#逐个相除
print(np.divide(x,y))

print(np.sqrt(x))
#逐元素开根

print("--------------矩阵向量乘法--------------")
"""
用dot表示向量和矩阵、矩阵和矩阵、向量和向量之间相乘
"""
x=np.array([[1,2],[3,4]])
y=np.array([[5,6],[7,8]])
#创建两个np数组

v=np.array([9,10])
w=np.array([11,12])
#另一组

print("np.dot(v,w) computes the dot product of v and w")
print(v.dot(w))
#或者等价地
print(np.dot(v,w))
#计算w和v的点积

print(x.dot(v))
#或者
print(np.dot(x,v))
#不满足交换律

print('----------ML/DL functions----------------')
print('max and argmax')
x=np.array([[1,-2,3],[-4,5,-6]])
"""
np.max操作指定维度的最大值，例如x.shape=(2,3)
按照第二个维度取最大值，压缩成(2,),得到[3,5]
如果希望返回最大值索引，而非数值，需要用argmax
这时候返回shape仍然是(2,)，得到[2,1]
"""
print(np.max(x,axis=1))
print('argmax return indexes instead of values')
print(np.argmax(x,axis=1))

print('np.clip(array,min,max) bounds all values in [min,max]')
"""
np.clip操作限制所有元素在[min,max]之间，小于min的变为min，大于max的变为max
不修改原数组，返回新数组
应该输出[[1,-2,3],[-3,3,-3]]
"""
print(np.clip(x,-3,3))


print('np.where(condition,x,y) returns a new array, with element=x if condition else y')
"""
np.where操作返回一个新数组，它的形状和原数组一样，其中condition=True则取x对应值
否则取y对应值，condition是一个条件表达式或者布尔数组

这里元素大于0返回本身否则返回0，打印出[[1,0,3],[0,5,0]]
np.where和ReLU激活函数有关
"""
print(np.where(x>0,x,0))

"""
np.exp(scores)基于广播机制，对每个元素做指数运算
"""
scores=np.array([2.0,1.0,0.1])
print(f"softmax: {np.exp(scores)/np.sum(np.exp(scores))}")

"""
np.linalg.norm(x,ord=None,axis=None)计算2范数
np.linalg.norm(x,ord=new_order,axis=new_axis)计算指定轴的指定范数
"""

print("----------------transpose & reshape-------------")
v=np.array([[1,2,3]])
print(v)
print("transpose of v:",v.T)
"""
这里v.shape=(1,3)，看作一行三列矩阵
转置之后v.shape=(3,1)，看作三行一列矩阵
"""
"""
reshape操作把x里所有数据，按照顺序填入新的大小一致的新数据规格当中
本质上改变数据的展现形式，不改变内容
"""
x=np.array([[1,2,3],[4,5,6]])
print(f"Original shape of x is {x.shape}")
print(f"flatten the array to new shape (dim,0),dim is the number of entries \n{x.reshape(-1)}")
print(f"Returns a copy of the flattened array\n{x.flatten()}")
print(x.reshape(6,1))
#reshaping x into the shape (6,1) with entries in original order
print(x.reshape(3,2))
#reshaping into (3,2) with entries in original order

print('--------------广播机制---------------')
"""
广播的三条规则（按顺序判断）
从最后一个维度开始向前比较（即从右往左看shape）。
如果两个维度相等，则继续比较前一个维度。
如果一个维度存在，一个维度不存在，就扩展到相同
如果其中一个维度为 1，则把它“拉伸”到和另一个维度相同。
如果两个维度既不相等，也不为 1，则无法广播，报错。
"""
a=np.array([1,2,3])
b=5
print("Sum of a and b is ",a+b)
"""
这里a.shape=(3,);b.shape=()
从右往左比较第一个维度：a维度是3，b没有维度，直接扩展成3
a=[1,2,3] b=np.array([5,5,5])  a.shape=(3,) b.shape=(3,)
逐个元素相加输出[6,7,8]
"""
a=np.array([[1,2,3],[4,5,6]]) #shape (2,3)
b=np.array([10,20,30])#shape (3,)
print("Sum of 2D a and 1D b is",a+b)
"""
这里比较最后一个维度，a和b都是3
然后看a的2，b没有维度，复制过去补一个维度2
a=[[1,2,3],[4,5,6]] #shape (2,3)
b=[[10,20,30],[10,20,30]] #shape (2,3)
a+b=[[11,22,33],[14,25,36]] #shape (2,3)
"""
print("sum with explicit loops")
x=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])#shape (4,3)
v=np.array([1,0,1])#shape (3,)
y=np.empty_like(x)#an empty matrix with the same shape as x

for i in range(4):
    y[i,:]=x[i,:]+v
print(y)
#显式写法，实现把v加到x的每一行
print("sum with broadcasting")
x=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])#shape (4,3)
v=np.array([1,0,1])#shape (3,)
y=x+v
print(y)
#广播写法,把x扩展成(4,3)

print("Compute outer products of vectors")
v=np.array([1,2,3])# shape(3,)
w=np.array([4,5])# shape (2,)
#把v的形状改成(3,1)即三行一列然后
#v和w广播之后，v.shape=(3,2)，w.shape=(3,2)
#v和w逐元素相乘
print(np.reshape(v,(3,1))*w)
#和下面的等价
print("outer product of v and w:",np.outer(v,w))

print("scaling a matrix")
print(x*2)
#标量直接被广播和x.shape一模一样，逐元素相乘

print("add a vector to rows of matrix")
x=np.array([[1,2,3],[4,5,6]])
v=np.array([1,2,3])
print(x+v)
#x两行三列，v一行三列，v广播到两行三列,每一行都是v，相当于x每一行加v

print("add a vector to columns of a matrix")
w=np.array([4,5])
print((x.T+w).T)
print("For 2D arrays transpose is same as zip")
#x.T三行两列数据,转置交换轴排列[[1,4],[2,5],[3,6]]
#w一行两列，广播到三行两列，相当于把w加到x.T的每一列，[[5,9],[6,10],[7,11]]
#转置回来[[5,6,7],[9,10,11]]

print("alternative for adding to columns is reshape into row vector")
print(x+np.reshape(w,(2,1)))
#w.shape=(2,1) x.shape=(2,3)
#broadcast-w.shape=(2,3) x.shape=(2,3)
#w_new=[[4,4,4],[5,5,5]]
#which equivalents adding w to each col

print('----------------向量化编程----------------')
print('-------replacing python loops with vectorized NumPy operations---------')
"""
图像作为nparrays：图片可以看成一个3D的numpy数组，形状是(height,width,3)
外层是一个二维数组，每一个数组元素都是一个3元素小数组，即RGB三通道
图像束(batches)作为nparrays：一个batch of images可以看成4维的数组，第一个数组就是图像自己的编号
剩下三个维度和图像一样，即(index,height,width,3)
"""
print("simulate a batch of 16 images,with 32*32 pixels each:")
images=np.random.default_rng(42).integers(0,256,size=(16,32,32,3),dtype=np.uint8)
print(f"shape of the batch of images: {images.shape}")# shape (16,32,32,3) 
print(f"shape of a single image: {images[0].shape}")# shape (32,32,3)
print(f"R channel : {images[0,:,:,0].shape}")# shape (32,32)
print(f"Mean RGB value: {images.mean(axis=(0,1,2))}")# avg over index.height.width
#这里是前三个维度全部压缩掉，剩下的是三通道分别的像素均值，结果形状就是(3,)

print("-----------------Matplotlib---------------------")
print("-----------------图像绘制和显示-------------------")
import matplotlib.pyplot as plt
x=np.arange(0,3*np.pi,0.1)
#构造点列
y_sin=np.sin(x)
y_cos=np.cos(x)
#y的表达式
plt.plot(x,y_sin)
plt.plot(x,y_cos)
#画图命令，x为横轴，y为纵轴
plt.xlabel('x axis label')
plt.ylabel('y axis label')
#标记横纵轴
plt.title('Sine and Cosine')
#图形标题
plt.legend(['Sine','Cosine'])
#图例命令
plt.show()
#打开窗口

x=np.arange(0,3*np.pi,0.1)
#构造点列
y_sin=np.sin(x)
y_cos=np.cos(x)
#y的表达式
plt.subplot(2,1,1)
#参数：行数、列数、当前子图编号
#将窗口分成二行一列，激活第一个子图(左上角第一个，从左到右从上到下进行编号)
#后续操作都会用在这个子图上
plt.plot(x,y_sin)
plt.title('Sine')
#第一个子图及其标题
plt.subplot(2,1,2)
#激活第二个子图，后续操作这个子图
plt.plot(x,y_cos)
plt.title('Cosine')
plt.show()
print("-------------图片显示---------------------")
#TODO
