#### 1.提出了什么问题：
- 用户行为的异质性
- 数据稀疏性
#### 2.解决方法：
- 双向物品-行为融合架构：在早期和中期两个阶段进行物品和行为信息的融合来增强行为的语义建模和捕捉多尺度的行为模式
- 三个层级行为数据增强方法：共现增广；基于频率的掩码；辅助行为翻转
#### 3.任务定义与符号：
U,V,B分别代表用户集合，物品集合以及行为集合
整个序列交互集合可以表示为$S_u = {(v_u^1,b_u^1),(v_u^2,b_u^2),...,(v_u^L,b_u^L)}$，L代表序列长度
使用多热向量$b_u^l = (b_{u,1}^l,...,b_{u,|B|}^l)$来表征行为集合的通用形式，$b_{u,k}^l$=1表示在序列的l位置进行的第k种交互
#### 4.整体框架：
先使用三种数据增强得到增强后的行为序列B，然后同时进行早期和中期的物品-行为融合，将结果$\check{U}$送入CrossAttn，然后进入FFN。
#### 5.各模块的关键内容：
双向物品-行为融合：
- embedding层，物品序列被编码为E∈$\mathbb{R}^{L\times d}$,可学习的位置编码P∈$\mathbb{R}^{L \times d}$,行为embedding矩阵为$G∈\mathbb{R}^{|B| \times d}$,对其进行用户偏好增强使其成为个性化的行为嵌入集合，$B_u^l = softmat(f_u ⊙ b_u^l) · G$,其中$F∈\mathbb{R}^{|U|\times{|B|}}$
- 早期的物品-行为融合，融合物品embedding和行为embedding，$E'=f(E,B)$,f()可以实例化为求和，拼接或者门控操作。使用一个多头Transfomer和逐点前馈网络FFN(·)，$X_e=E'+P,F=FFN(MHAttn(X_e)),F∈\mathbb{R}^{L\times d}$
- 中期物品-行为融合:(a)BASA模块，输入为X=E+P，将X与B转换为QK的查询矩阵后再求和，$A_h=Q_h^X(K_h^X)^T+Q_h^B(K_h^B)^T$，然后得到结果为$R_h=(softmax(\dfrac{A_h}{\sqrt{d_h}})⊙∆)V_h^X$,其中∆为因果掩码的下三角矩阵，所有头的结果拼接起来得到R。 (b)BGMoE模块以行为嵌入B和BASA的输出R作为输入$O=\sum_{i=1}^n{φ(B)_ie_i(R)},φ(·)$通过线性映射接softmax的归一化操作计算路由权重，$e_i$表示使用FFN(·)的第i个专家系统。
- 用户表征，将早期和中期融合特征作为输入，$\check{U}=αO+(1-α)F,\check{U}∈\mathbb{R}^{L\times d},$α是超参，为了使用户偏好与下一步行为集合的语义对齐，构建下一步行为集合$T=[B^2,B^3,...,B^{L+1}]$,然后将其作为Q，$\check{U}$作为K和V向量计算最终的表征$U=FFN(CrossAttn(T,\check U))$,CrossAttn参照上文公式
行为层级的数据增强模块：
- 更关注用户行为序列而不是物品序列。三种方法都能够感知频率，倾向于添加低频行为并遮蔽高频行为。
- 共现行为增广，p=b · M，b是行为序列，$M∈\mathbb{R}^{|B|\times{|B|}}$代表物品共现矩阵。根据概率p$$p \leftarrow \dfrac{p}{\sum_{k=1}^{|B|}p_k} $$将共现行为添加回行为序列b。
- 基于频率的掩码设置，先计算行为频率向量$m∈\mathbb{R}^{|R|}$,掩码概率为$$P(b_i=0)=\dfrac{m_i^c}{\sum_{k=1}^{|B|}{m_k^c}}$$c是为了平滑高频行为的超参数。
- 辅助行为翻转，同样按照给定的概率随机将一些辅助行为交互进行翻转，即交互与未交互进行翻转$b_{b_a}=1-b_{b_a}$
计算目标和损失函数设计：
- 对于用户u的下一个行为预测，$\hat{y}_{l+1,v}=u_le_v^T$,损失函数设计
-  下一物品预测损失 $\mathcal{L}_{next}$ 采用 **BCE 损失**做 next-item prediction,负样本从 $\mathcal{V}\setminus S_u$ 中随机采样。  **行为丰富度加权**:目标行为集合包含多种行为类型的预测步,损失权重更高。 $$w_{u,l}=\frac{\lVert b_{u,l+1}\rVert_0}{|B|}$$ $\delta(\cdot)$ 为指示函数:真实物品为 1,padding 为 0;$|\delta(v)|$ 为所有序列中有效真值物品总数,用于归一化。  
- 序列级对比损失 $\mathcal{L}_{SeqCL}$ - 对同一用户序列 $S_u$ 生成两个增强视图,分别编码为 $H_u^1,H_u^2\in\mathbb{R}^{L\times d}$。将各步表示拼接为 $h_u^1,h_u^2\in\mathbb{R}^{L\cdot d}$。 
- 对称形式:$\mathcal{L}_{SeqCL}=\mathcal{L}_{CL}(h^1,h^2)+\mathcal{L}_{CL}(h^2,h^1)$ 
- 相似度为**点积**,负样本为当前 mini-batch 内其他序列的增强表示,τ 为温度参数。
#### 6.实验设置：
### 数据集 
| 数据集 | 物品 | 辅助行为 | 目标行为 |
|---|---|---|---|
| KuaiSAR(推荐子集) | 短视频 | click | like, share, follow |
| QK-Article(Tenrec) | 文章 | read | like, share, favorite, follow |
| QK-Video(Tenrec) | 短视频 | click | like, share, follow |
### 评估
- 指标:HR@k、NDCG@k,k ∈ {5, 10} 
- **全排序(full-ranking)**设置
### 参数设置
- 公平对比:所有模型 d = 32,序列长度 L = 50,使用相同的截断序列。 
- 早期与中期融合的堆叠块数:{2, 3} 
- 注意力头数:{2, 4, 8} 
- dropout:{0.2, 0.3, 0.4, 0.5} 
- BGMoE 专家数:{4, 6, 8}