
#### 章节安排


1. 背景介绍
2. 均方根误差MSE
3. 最小二乘法
4. 梯度下降
5. 编程实现


## 背景




---


生活中大多数系统的输入输出关系为线性函数，或者在一定范围内可以近似为线性函数。在一些情形下，直接推断输入与输出的关系是较为困难的。因此，我们会从大量的采样数据中推导系统的输入输出关系。典型的单输入单输出线性系统可以用符号表示为：


y\=f(x)\=kx\+b其中，k为斜率，反应了当输入量x变化时，输出y的变化与输入x变化的比值；b反应了当系统没有输入（或输入为0）时，系统的输出值。


数据一般称**观测数据**或**采样数据**，这两种说法具有一定的侧重点，**观测**倾向于客观系统，例如每天的涨潮水深；**采样**倾向于主观系统，例如，对弹簧施加10N的压力，观察弹簧的形变量。


对于但输入单输出系统，数据可以表示为：


O\={oi}N\={xi,yi}N或


S\={si}N\={xi,yi}N其中符号O对应**observation(观测)**、符号S对应**sampling(采样)**,{oi}N中oi表示采样序列中的每一个元素，N表示序列中元素的个数，xi表示系统输入，yi表示系统输出


在系统的推导过程中，一般称推导的结果为对实际系统的估计或近似，用符号记为y^\=f^(x)。对于单个采样点，系统的误差定义为：对该采样输入，输出的真实值与输出的预测值的差为误差。用数据公式表示为：


εi\=yi−yi^\=yi−f^(xi)对于整体采样序列，一种经典的误差是**均方根误差**（Mean Squared Error, **MSE**），其数学公式为：


MSE\=∑i\=1Nεi2在推导系统输入输出关系，通常有两种方法，一种是基于数值推导的方法，一种是基于学习的方法。本文分别以最小二乘法和梯度下降为例讲解两种方法。


## MSE


对于单个采样点的情形，MSE退化为方差的平方，即：


MSE\=ε2\=(y−y^)2假定参数b为常量，仅考虑MSE与参数的关系，有


ε2\=(kx\+b−y)2\=x2(k\+b−yx)2易得，MSE是关于k的二次函数，且该二次函数有唯一的零点：k0\=−(b−y)/x


对于多个点的情形，对每个点{si}\={xi,yi}，εi2均可表示为关于k的二次函数，有：


MSE\=∑i\=0Nεi2\=∑i\=0N(xi2(k\+b−yixi)2)\=∑i\=0N(aik2\+bik\+ci)\=Ak2\+Bk\+C即：序列的MSE也为关于参数k的二次函数，并且，MSE≥0，当且仅当(b−yi)/xi\=M为常数时不等式取等。



> 可以很容易证明MSE也是关于参数b的二次函数


开口向上的二次函数有两个重要的性质：


1. 导数为0的点，为其最小值点。


f(xi)\=minf(x)⟺f′(xi)\=02. 任意点距离最小值点的距离与其导数值成正比，方向为导数方向的反方向


xi−xmin∝−f′(xi)性质1、2分别是最小二乘法、梯度下降法的理论基础/依据。


## 最小二乘法




---


最小二乘法基于MSE进行设计，其思想为，找到一组参数，使得MSE关于每个参数的偏导为0，对于一元输入的情形，即：


(3\.1\)∂MSE∂k\=0(3\.2\)∂MSE∂b\=0首先化简公式(3\.2)


∂MSE∂b\=1N∑i\=1N∂(εi2)∂b\=1N∑i\=1N2ϵi⋅∂∂b(εi)\=2N∑i\=1Nϵi⋅∂∂b(kxi\+b−yi)\=2N∑i\=1N(kxi\+b−yi)\=2N(k∑i\=1Nxi\+Nb−∑i\=1Nyi)由公式(3\.2)有：


2N(k∑i\=1Nxi\+Nb−∑i\=1Nyi)\=0(3\.3\)b\=1N(∑i\=1Nyi−k∑i\=1Nxi)其次化简公式3\.1


∂MSE∂k\=1N∑i\=1N∂(εi2)∂k\=1N∑i\=1N2ϵi⋅∂∂k(εi)\=2N∑i\=1Nϵi⋅∂∂k(kxi\+b−yi)\=2N∑i\=1Nxi(kxi\+b−yi)\=2N(k∑i\=1Nxi2\+b∑i\=1Nxi−∑i\=1Nxiyi)代入公式(3\.1),(3\.3)有：


2N(k∑i\=1Nxi2\+b∑i\=1Nxi−∑i\=1Nxiyi)\=0k∑i\=1Nxi2\+1N∑i\=1Nxi(∑i\=1Nyi−k∑i\=1Nxi)−∑i\=1Nxiyi\=0k(∑i\=1Nxi2−1N(∑i\=1Nxi)2)\=∑i\=1Nxiyi−1N∑i\=1Nxi∑i\=1Nyi(3\.4\)k\=N∑xi2−(∑xi)2N∑xiyi−∑xi∑yi公式(3\.3),(3\.4)即为最小二乘法的参数公式


### 梯度下降




---


对于学习机器学习的初学者，我们首先讨论最简单的情形：基于单个采样点的学习。


二次函数具有重要性质：任意点距离最小值点的距离与其导数值成正比


xi−xmin∝−f′(xi)基于该性质，我们可以可以设计参数更新公式如下


Δkt\=−λ∂εi2∂k\=−λ(2εi∂εi∂k)\=−λ(2εixi)Δbt\=−λ∂εi2∂b\=−λ(2εi∂εi∂b)\=−λ(2εi)故有参数更新公式：


(4\.1\)εi\=y−(kxi\+bi)(4\.2\)k:\=k−λ(2εixi)(4\.3\)b:\=v−−λ(2εi)其中λ为学习率，一般取0\.1∼10−6



> 常数2是可以缺省的，可以视为学习率放大了两倍。


## 编程实现



> 建议读者按照如下方法创建头文件、定义函数
> `typedef.h` ：定义变量类型
> `random_point.h`：生成随机点
> `least_square.h`：最小二乘法的实现
> `gradient_descent.h`：梯度下降方法的实现


### 类型定义




---


首先我们需要定义采样点，以及采样点序列类型。
采样点是包含x、y两个值的数据类型。同时，为方便使用，定义别名`Point`
采样点序列，或者称数据，可以存储为类型为`Point`的`vector`



```


|  | struct SamplePoint{ |
| --- | --- |
|  | float x; |
|  | float y; |
|  | } |
|  | using Point = SamplePoint; |
|  |  |
|  | using Data = std::vector; |


```

对于直线，其包含k，b两个参数，同时，为了方便调用，定义括号运算符`()`重载



```


|  | struct LinearFunc{ |
| --- | --- |
|  | float k; |
|  | float b; |
|  | float operator()(float x){ |
|  | return k*x+b; |
|  | } |
|  | } |
|  | using Line = LinearFunc; |
|  | using Func = LinearFunc; |


```

### 数据生成




---


采用`random`库中的`normal_distribution`随机数引擎



```


|  | #include |
| --- | --- |
|  | #include |
|  | #include "typedef.h" |
|  |  |
|  | Data generatePoints(const Func& func, float sigma, float a, float b, int numPoints) { |
|  | Data points; |
|  | std::random_device rd; |
|  | std::mt19937 gen(rd()); |
|  | // std::uniform_real_distribution<> distX(a, b); // 均匀分布 |
|  | std::normal_distribution<> distX((a + b) / 2, (b - a) / 2.8); // 正态分布 |
|  | std::normal_distribution<> distY(0, sigma); |
|  |  |
|  | for (int i = 0; i < numPoints; ++i) { |
|  | float x = distX(gen); |
|  | float y = func(x) + distY(gen); |
|  | points.push_back({ x, y }); |
|  | } |
|  |  |
|  | return points; |
|  | } |


```

该方法接受五个输入，分别是：


1. `func`：函数，自变量x与自变量y的关系
2. `sigma`：y的观测值与真实值的误差的方差
3. `a`、`b`：生成的数据范围的参考上下界，决定了生成数据的宽度，同时，绝大多数数据将位于此区间
4. `numPoints`：点的个数


### 最小二乘法




---


最小二乘法仅需接受一个输入：数据`Data`，同时返回数据。


(3\.4\)k\=N∑xi2−(∑xi)2N∑xiyi−∑xi∑yi(3\.3\)b\=1N(∑i\=1Nyi−k∑i\=1Nxi)在实现中，需要遍历采样数据，并分别进行累加计算∑xi、∑yi、∑xi2和∑xiyi



```


|  | Line Least_Square(const Data& data) { |
| --- | --- |
|  | Line line; |
|  |  |
|  | float s_x = 0.0f; |
|  | float s_y = 0.0f; |
|  | float s_xx = 0.0f; |
|  | float s_xy = 0.0f; |
|  |  |
|  | float n = static_cast<float>(data.size()); |
|  |  |
|  | for (const auto& p : data) { |
|  | s_x += p.x; |
|  | s_y += p.y; |
|  | s_xx += p.x * p.x; |
|  | s_xy += p.x * p.y; |
|  | } |
|  |  |
|  | line.k = (n * s_xy - s_x * s_y) / (n * s_xx - s_x * s_x); |
|  | line.b = (s_y - line.k * s_x) / n; |
|  |  |
|  | return line; |
|  | } |


```

### 梯度下降




---


梯度下降法是一种学习方法。对参数的估计逐渐向最优估计靠近。在本例中表现为，MSE逐渐降低。
首先实现单步的迭代，在该过程中，遍历所有的采样数据，依据参数更新公式对参数进行修正。


(4\.1\)εi\=y−(kxi\+bi)(4\.2\)k:\=k−λ(2εixi)(4\.3\)b:\=v−−λ(2εi)梯度下降法需要一个给定的初值，对于线性函数，除了人工生成、随机初值外，一种方式是，假定为正比例函数，以估计k，假定为常函数，以估计b，公式如下：


(5\.1\)k0\=∑yi/∑xi(5\.2\)b0\=∑yi/N在本例中，设定为对初值进行100次迭代后得到最终估计，读者可根据实际情况调整，在学习度设计的合适的情况下，一般迭代次数在50∼200次



```


|  | #include "typedef.h" |
| --- | --- |
|  |  |
|  | constexpr float eps = 1e-1; |
|  | constexpr float lambda = 1e-5; |
|  |  |
|  | void GD_step(Func& func, const Data& data) { |
|  | for (const auto& p : data) { |
|  | float error = func(p.x) - p.y; |
|  | func.k -= lambda * error * p.x; |
|  | func.b -= lambda * error; |
|  | } |
|  | } |
|  |  |
|  | Func Gradient_Descent(Func& func, const Data& data) { |
|  | float s_x = 0, s_y = 0; |
|  | for (const auto& p : data) { |
|  | s_x += p.x; |
|  | s_y += p.y; |
|  | } |
|  |  |
|  | Line line; |
|  | line.k = s_y / s_x; |
|  | line.b = s_y / data.size(); |
|  |  |
|  | float lambda = 1e-5f; |
|  |  |
|  | for (size_t _ = 0; _ < 100; _++) { |
|  | GD_step(line, data); |
|  | } |
|  |  |
|  | return line; |
|  | } |


```

## 附录


### nan问题


该问题有两种产生的原因，参数更新符号错误及学习率过高。


**参数更新符号错误**
在更新公式中，如果错误的使用\+号，或者采用y^−y计算εi，都将会导致参数向误差更大的方向更新，经过了数次迭代后，与真实值的距离越来越远，最终产生nan。


k:\=k−λ(2εixi)**学习率过高**
如下图，当学习率设置的过高时，新的参数组{kt\+1,bt\+1}将比旧参数{kt,bt}带来更大的估计误差（红色箭头），而良好的学习率是使得估计误差逐渐下降的
![description](https://img2024.cnblogs.com/blog/3320410/202411/3320410-20241119181435639-1469528137.png)


 本博客参考[MeoMiao 萌喵加速](https://biqumo.org)。转载请注明出处！
