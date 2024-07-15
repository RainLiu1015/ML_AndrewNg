# 1 Linear Regression With One Variable 

## 1.1 Example

- fitting a straight line with data

<img src="/Users/apple/Desktop/Code/MLwuenda/notes/1Supervised-1.assets/截屏2024-07-15 12.58.26.png" alt="截屏2024-07-15 12.58.26" style="zoom: 33%;" /><img src="/Users/apple/Desktop/Code/MLwuenda/notes/1Supervised-1.assets/截屏2024-07-15 13.00.46.png" alt="截屏2024-07-15 13.00.46" style="zoom: 33%;" />

- Regression model: **output specific numbers**-> infinite possible output

## 1.2 Terminology

- Training set
  - $x =$ input variable feature
  - $y = $output variable/target variable
  - $(x, y) = $single training example
  - $m=$number of training example
  - $(x^{(i)}, y^{(i)}) = i^{th}$ training example  
- Test set
- Function: $f(x) = \hat{y}$，其中$\hat{y}$为prediction，与$y$即真实output是不同的
- **cost function**

## 1.3 Cost Function

在linear regression中，我们将$f$表示为$f_{w, b} = wx + b$ —— **univariate linear regression**

### 1.3.1 什么是Cost Function

cost function用来表示当前model的表现，cost越少表示model越准确

- error: $(\hat{y} - y)$
- $J(w, b) = \frac{1}{2m}\sum_{i = 1}^m (\hat{y^{(i)}} - y^{(i)})^2$: Square error cost function(most widely used)
- 相同的，$J(w, b) = \frac{1}{2m}\sum_{i = 1}^m (f_{w, b}(x^{(i)}) - y^{(i)})^2$

**goal：minimize $J(w, b)$**

### 1.3.2  Visualization

#### 简化条件：假设$b = 0$

在一个简单的linear regression例子中，cost function的图像可能看起来像这样： <img src="/Users/apple/Desktop/Code/MLwuenda/notes/1Supervised-1.assets/截屏2024-07-15 13.56.41.png" alt="截屏2024-07-15 13.56.41" style="zoom:25%;" />此时b=0。

显然此时需要选择合适的w使得J达到最小值。

#### 真实条件：w和b都是变量

<img src="/Users/apple/Desktop/Code/MLwuenda/notes/1Supervised-1.assets/截屏2024-07-15 14.02.04.png" alt="截屏2024-07-15 14.02.04" style="zoom: 33%;" />

此时的cost function图像看起来如上，为了在二维平面中可以表示，我们常用一个contour plot（等高线图）表示它：

<img src="/Users/apple/Desktop/Code/MLwuenda/notes/1Supervised-1.assets/截屏2024-07-15 14.05.15.png" alt="截屏2024-07-15 14.05.15" style="zoom:33%;" />

在真实的ML方法中，不需要真的通过contour plot寻找w和b。

## 1.4 Gradient Descent

### 1.4.1 什么是gradient descent

gradient descent 适用于很多情景，而不仅限于减小cost function。

步骤：

- 任意选择一个起始位置，一般是$w = b = 0$
- 持续改变$w$和$b$，直到达到一个足够满意的结果
- 在较为复杂的情况中，这实际上是一个优化问题，我们能够找到的也只local optimal

### 1.4.2 algorithm

$w = w - \alpha \frac{\partial}{\partial w}J(w, b)$

- 不断改变w的值，以得到更优的答案
- $\alpha$：learning rate，通常是0-1之间的一个很小的值，由于控制优化的**步长**->如何选择一个好的learning rate？
- $\frac{\partial}{\partial w}J(w, b)$：实际上决定了优化的**方向**

类似的：$b = b - \alpha \frac{\text{d}}{\text{d}b}J(w, b)$

几个注意点：

- 我们的目标是收敛（convergence）

- 我们需要同时update $w$和$b$:
  $$
  tmpw =w - \alpha \frac{\partial}{\partial w}J(w, b)\\
  tmpb =b - \alpha \frac{\partial}{\partial b}J(w, b)\\
  w = tmpw, b = tmpb
  $$
  即需要先分别进行update，然后赋值，不能让先update的影响到未update的那个variable。

### 1.4.2 Learning Rate

- 当learning rate太小时，descent速度会很慢，需要更多次的计算和更长的时间
- 当learning rate太大时，descent可能会出错：错过真正的optimal点，甚至可能永远无法到达——overshoot/diverge

如果当前的J已经在local minimal时，还会update吗？不会，偏导是0——此时的初始值选择非常重要！

#### fix learning rate

- 当我们靠近local minimum时，导数变小，导致update量减少，descent速度减慢

## 1.5 Gradient Descent for Linear Regression

  推导：
$$
\begin{aligned}
\frac{\partial }{\partial w}J(w, b) 
& = \frac{\partial }{\partial w} \frac{1}{2m}\sum_{i = 1}^m (wx^{(i)} + b - y^{(i)})^2\\
& = \frac{1}{2m} \sum_{i = 1}^m  2x^{(i)}(wx^{(i)} + b - y^{(i)})\\
& = \frac{1}{m} \sum_{i = 1}^m  x^{(i)}(wx^{(i)} + b - y^{(i)})\\
& = \frac{1}{m} \sum_{i = 1}^m  x^{(i)}(f_{w, b}(x^{(i)})- y^{(i)})\\
\end{aligned}
$$

$$
同理，有\frac{\partial }{\partial b}J(w, b)  =  \frac{1}{m} \sum_{i = 1}^m (f_{w, b}(x^{(i)})- y^{(i)})\\
$$

所以我们有：
$$
\begin{aligned}
重复&\\
& w = w - \alpha\frac{1}{m} \sum_{i = 1}^m  x^{(i)}(f_{w, b}(x^{(i)})- y^{(i)})\\
& b = b - \alpha\frac{1}{m} \sum_{i = 1}^m (f_{w, b}(x^{(i)})- y^{(i)})\\
直到收敛
\end{aligned}
$$

- 注意要**同时update** $w$和$b$。

- 在linear regression中，J是$w$和$b$的凹函数，也就是说不存在多个local minimum；于是最后的update**一定会converge**。

### 1.5.1 Batch Gradient Descent

- Batch：Each step of gradient descent uses all the training examples

