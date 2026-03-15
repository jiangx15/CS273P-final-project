## 内容简要说明

把具体在干嘛简单说一下，细节不知道直接发群里问。report结构应该在canvas上也有说，可以看着来。
直接看[`demo.ipynb`](demo.ipynb)也行，里面基本上包含整个流程。
---
本项目使用 **Bank Marketing Dataset** 预测银行客户是否会订阅定期存款（binary classification）。它包括三个模型：

- 逻辑回归
- MLP
- Transformer

目标列是“y”，其中“yes -> 1”和“no -> 0”。
---
### 1. Introduction
介绍项目研究的问题，例如：

- 银行营销任务背景
- 为什么预测客户是否会订阅定期存款是一个有意义的问题
- 机器学习如何帮助银行提高营销效率

可以简单说明：

- 银行营销成本较高
- 需要预测哪些客户更可能订阅
- 使用机器学习模型进行预测
---

### 2. Dataset（数据集）

数据集：

<https://www.kaggle.com/datasets/sushant097/bank-marketing-dataset-full>

可以介绍：

- 数据来源（Kaggle）
- 数据规模
- 特征类型

例如：

- 人口信息（age, job, marital）
- 金融信息（balance, loan）
- 营销相关信息（campaign, contact, duration）

另外可以说明：

- 数据集是 **class imbalance** 的
- 正类（订阅）比例较低

`demo.ipynb` 中已经包含了数据分布的可视化。

---

### 3. Methods（方法）

这一部分介绍使用的模型。

本项目使用了三个模型：

- Logistic Regression
- MLP
- TabTransformer

其中：

- 前两个作为 **baseline**
- TabTransformer 作为 **主要模型**

具体实现代码位于：src/


#### 1.Logistic Regression

作为传统机器学习 baseline，用于对比深度学习模型。


#### 2.MLP
使用 embedding + fully connected layers 的神经网络模型，用于处理 tabular data。


#### 3.TabTransformer（主要模型）
使用：

- categorical embedding
- Transformer encoder
- attention mechanism

用于学习不同特征之间的关系。
---

### 4. Training Setup（训练设置/实验设置）

介绍模型训练方法，例如：

- optimizer：Adam
- learning rate：1e-3
- batch size：256
- epochs：30
- loss function：Binary Cross Entropy

具体配置在：configs/

数据划分：
```
Train / Validation / Test
70% / 15% / 15%
```
---

### 5. Experiments result（实验结果）
实验结果可以直接参考：demo
Notebook中已经包含运行后的结果和可视化，他应该打开就有我跑过的缓存。

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

其实大概意思就是，比如你跟一个人推销银行的这个存款套餐，假说他愿意听你讲还讲了巨久那基本上他也会存的，相当于这个特征指向性很强，但如果真实预测场景中，我们通常 **在拨打电话之前就需要进行预测**，所以对这个特征进行消融实验。

---

#### 3️⃣ 可视化结果

对主模型：
- ROC Curve
- Confusion Matrix

比较模型之间：

- ROC-AUC comparison
- F1 comparison

---

### 6. Results & Discussion（结果分析）

可以讨论：

- TabTransformer 是否优于 baseline
- duration 对模型性能的影响

例如：

- 为什么包含 duration 时模型效果明显提升
- 去掉 duration 后模型表现变化

---

### 7. Conclusion（总结）
总结项目主要结论，例如：

- 不同模型的表现
- 关键特征的影响
- 模型的局限性

可以简单讨论未来改进方向。
