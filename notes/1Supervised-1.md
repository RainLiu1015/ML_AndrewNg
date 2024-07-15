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

