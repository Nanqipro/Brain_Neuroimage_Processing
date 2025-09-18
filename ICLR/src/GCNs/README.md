## 代码说明
1. 这是基于原来 GCN 代码的规整化版本
2. model.py: 封装好了 4 种模型：
+ GCN: PureGCN
+ GraphSAGE: PureGraphSAGE
+ GAT: PureGAT
+ Hybrid: ImprovedGCN（这是我们 Workshop 的论文模型，可以忽略）
3. process.py: 数据处理的一系列方法
+ 预处理：可以不修改，但二分类可能会报错，SMOTE 源码我没有仔细看
+ 图生成：根据我们的图数据结构进行重构，生成对应的 PyG 数据对象，便于训练，即 `generate_graph`, `create_pyg_dataset`
4. train.py: 对应的训练和测试方法
5. run.py: 运行模型的启动函数以及结果分析，仔细查看 `parse_args` 中的参数，并对超参数根据数据做出一些调整

## To-do list
1. 对图数据做适配：txt(../../data/graphs) - PyG - tensor
2. 注意这是监督学习，每个图都有 label, ../../data/labels
3. 做多组实验（50-100组），注意数据集的划分
4. 指标：每个模型的 accuracy, recall, f1-score, training time, predicting time, run time(纯数据就好，图我们自己另外画，数据处理无需记录时间)
5. 找其他论文的模型（SOTA），重复以上的步骤，Paper with code / Google Scholar
6. 注意看是单卡还是多卡，如果爆显存（OOM）了请记录