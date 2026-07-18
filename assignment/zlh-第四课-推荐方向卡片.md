# 检索方式 
google scholar高级检索，关键词包括**Multimodal Recommendation**，**Contrastive Learning**，检索条件为包括全部关键词。选择被引数最高的相关论文**Multimodal Graph Contrastive Learning for Multimedia-Based Recommendation**
# 术语
Separated Graph Learning Module 分离式图学习模块
Node Self-discrimination Task 节点自判别任务
Message Propagation 消息传播
Collaborative Signals 协同信号
Modality-specific User Preference Clues  模态特定用户偏好线索
Multimodal Noise Contamination 多模态噪声污染
Weight Transformation Matrix 权重变换矩阵
Popularity-aware Norm 流行度感知归一化
Over-smoothing 过平滑
Bayesian Personalized Ranking (BPR) Loss 贝叶斯个性化排序损失
Matrix Factorization (MF) 矩阵分解
Collaborative Filtering (CF) 协同过滤
Disentangled Representation Learning 解耦表示学习
# 方法
本文提出了一种名为 **MGCL (Multimodal Graph Contrastive Learning)** 的框架，旨在解决多媒体推荐中的噪声污染问题。
**分离式图学习模块** 构建三个并行的图卷积网络（GCN），分别在用户-物品交互图 ( GG )、视觉交互图 ( GvGv​ ) 和文本交互图 ( GtGt​ ) 上独立进行消息传播，生成三种类型的用户和物品表示（协同信号、视觉偏好线索、文本偏好线索）。
**对比损失 (Contrastive Loss)** 引入节点自判别任务，假设同一节点在不同偏好下的表示相似度应高于不同节点在同一偏好下的相似度。利用 SimCLR 框架构建用户和物品的对比损失，以消除与偏好无关的噪声信息。
**交替训练策略 (Alternative Training)** 对推荐任务（BPR Loss）和对比学习任务（Contrastive Loss）进行交替优化，而非联合训练。
# 任务或数据形式
运用在多媒体推荐领域，具体任务为Top-K推荐/排序任务
使用数据集： 共使用了 3 个公开数据集，均采用 5-core 设置（保留至少5次交互的用户和物品）。 
- Beauty: Amazon 产品数据，包含图像和标题。 
- Art (Arts Crafts and Sewing): Amazon 产品数据，包含图像和标题。 
- Taobao: 天池竞赛数据集，仅包含视觉内容（无文本）。
# 评价指标
Hit Ratio (HR@k)
Normalized Discounted Cumulative Gain (NDCG@k)
