# MobileMamba的神经架构搜索(NAS)框架

此目录包含一个完整的神经架构搜索框架，专门为MobileMamba项目中的StarNet_NEW_CONV模型设计。

## 概述

NAS框架提供了自动搜索StarNet_NEW_CONV模型空间内最优神经网络架构的工具。它包括：

1. **搜索空间定义** - 定义可优化的参数
2. **搜索算法** - 用于探索搜索空间的多种算法
3. **性能评估** - 评估架构性能的方法
4. **工具函数** - 用于保存/加载配置和结果的辅助函数
5. **控制器** - 运行NAS实验的主要接口

## 目录结构

```
NAS/
├── __init__.py              # 包初始化
├── search_space.py         # StarNet_NEW_CONV的搜索空间定义
├── search_algorithms.py    # 各种搜索算法的实现
├── evaluator.py            # 模型性能评估方法
├── utils.py                # 用于保存/加载的工具函数
├── controller.py           # 主NAS控制器
├── main.py                 # 命令行界面
└── README.md              # 本文件
```

## 搜索空间

StarNet_NEW_CONV的搜索空间包括：

- **模型维度**: 不同的通道配置 `[dims]`
- **模型深度**: 每个阶段的块数 `[depths]`
- **MLP比率**: 前馈网络中的扩展比率
- **小波类型**: 不同的小波变换类型
- **可学习小波**: 是否使用可学习的小波滤波器

## 搜索算法

1. **随机搜索** - 从搜索空间中随机采样架构
2. **进化搜索** - 使用带有变异和交叉的遗传算法
3. **贝叶斯优化** - (简化实现) 使用概率建模

## 使用方法

### 基本用法

使用默认参数运行NAS：

```bash
python NAS/main.py --algorithm random
```

### 高级用法

使用自定义参数运行进化搜索：

```bash
python NAS/main.py --algorithm evolutionary --population_size 30 --generations 15 --mutation_prob 0.3
```

运行贝叶斯优化：

```bash
python NAS/main.py --algorithm bayesian --max_evaluations 100
```

### 比较算法

比较不同搜索算法的性能：

```bash
python NAS/main.py --compare_algorithms
```

### 基准测试特定架构

对保存的架构配置进行基准测试：

```bash
python NAS/main.py --benchmark path/to/architecture.json
```

## 自定义

### 添加新搜索算法

1. 在 `search_algorithms.py` 中实现新类
2. 在 `controller.py` 中为新算法添加支持
3. 更新 `main.py` 中的命令行界面

### 修改搜索空间

1. 编辑 `search_space.py` 中的搜索空间定义
2. 相应地更新采样和编码/解码方法

### 自定义评估指标

1. 修改 `evaluator.py` 中的评估方法
2. 根据需要添加新的评估函数

## 结果

结果保存在实验目录中，包括：

- 最佳架构配置 (JSON)
- 搜索历史 (Pickle)
- 基准测试结果 (JSON)
- 进度日志 (文本)

## 架构配置示例

```json
{
  "dims": [40, 80, 160, 320],
  "depths": [1, 2, 4, 5],
  "mlp_ratio": 2.0,
  "wt_type": "db1",
  "learnable_wavelet": true
}
```

## 依赖项

- PyTorch
- THOP (用于FLOPs计算)
- NumPy

## 未来改进

1. 使用高斯过程实现完整的贝叶斯优化
2. 添加基于强化学习的搜索算法
3. 支持多目标优化(准确率vs效率)
4. 添加分布式计算支持以进行并行评估
5. 为较差的架构实现早期停止机制
