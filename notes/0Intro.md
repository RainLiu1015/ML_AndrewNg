# 0. Intro

## 0.1 What's ML?

> being able to give solution without specific programming. 

- supervised learning（most majority）

  - input X->output label Y（**X-Y mapping**）

  - traing with input X and **label**(correst answer) Y

  - 主要特征是训练数据时需要给出正确的labelY（相应的，data labeling是一件会消耗大量人力物力的事情）

  - Eg：Regression，预测无限多数字中的任意一个

  - Eg：Classification，预测所有当前case属于所有class中的哪一个，输出种类是有限的  

    <img src="/Users/apple/Desktop/Code/MLwuenda/notes/0Intro.assets/截屏2024-07-15 12.41.54.png" alt="截屏2024-07-15 12.41.54" style="zoom:25%;" />你可能需要多种类型的输入得到decision boundary

- unsupervised learning

  - Data没有label，需要algorithm自己寻找规律(structure, pattern)
  - Eg：Clustering（gene microarray, grouping customers）
  - Anomaly detection
  - dimensionality reduction(data compression)

- reinforcement learning（less mentioned）