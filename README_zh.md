## 内容简要说明

把具体在干嘛简单说一下，细节不知道直接发群里问。report结构应该在canvas上也有说，可以看着来。
直接看[`demo.ipynb`](demo.ipynb)也行。
---
该项目预测银行客户是否会在银行营销数据集上订阅定期存款。它包括三个模型：

- 逻辑回归
- MLP
- Transformer

目标列是“y”，其中“yes -> 1”和“no -> 0”。

### 1. Introduction


---

### 2. Dataset（数据集）

<https://www.kaggle.com/datasets/sushant097/bank-marketing-dataset-full>
链接里介绍都有，可以说一下他比较不平衡，demo里面也可以看到他不平衡的图。

---

### 3. Methods（方法）

这一部分介绍我们使用的模型, 前两个是baseline，最后那个transformer是主要的。模型具体实现都在src/底下。


#### 1.Logistic Regression

作为传统机器学习 baseline，用于对比深度学习模型。


#### 2.MLP



#### 3.TabTransformer（主要模型）

---

### 4. Training Setup（训练设置/实验设置）

介绍模型训练方法，例如：

- optimizer：Adam
- learning rate：1e-3
- batch size：256
- epochs：30
- loss function：Binary Cross Entropy
具体内容在configs/下。

数据划分：
```

Train / Validation / Test
70% / 15% / 15%

```
---

### 5. Experiments result（实验结果）
数据在demo那个地方有不用自己跑，他应该打开就有我跑过的缓存。

#### 1️⃣ Model Comparison

比较不同模型表现：

| Model | Accuracy | F1 | ROC-AUC |
|------|---------|----|--------|
| Logistic Regression | ... | ... | ... |
| MLP | ... | ... | ... |
| TabTransformer | ... | ... | ... |

---

#### 2️⃣ Ablation Study

我们进行了 **feature ablation experiment**：

比较：

```

with duration
without duration

```

原因：

`duration` 是通话时长，通常是一个非常强的特征，但在实际预测时可能不可用。

其实大概意思就是，比如你跟一个人推销银行的这个存款套餐，假说他愿意听你讲还讲了巨久那基本上他也会存的，相当于这个特征指向性很强，但如果实际应用中的时候我们是还没跟人打电话，所以对这个特征进行消融实验。

---

#### 3️⃣ 可视化结果

对主模型：
- ROC Curve
- Confusion Matrix
- Model performance comparison

比较模型之间：

- ROC-AUC comparison
- F1 comparison

---

### 6. Results & Discussion（结果分析）

这一部分需要解释实验结果，例如：

- TabTransformer 是否优于 baseline
- duration 对性能的影响

可以讨论：

- 为什么包含 duration 时模型效果更好
- 去掉 duration 后模型表现变化

---

### 7. Conclusion（总结）

