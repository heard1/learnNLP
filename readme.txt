知识图谱过程
	分词
	向量化
		word2vec
		tfidf
		elmo
		bert
	特征提取
	任务：分类
		NER
		关系抽取


nlp-机器学习-知识图谱
123456789
10 11

朴素贝叶斯
逻辑回归
支持向量机
HMM
CRF
K-Means
情感分析
关键词提取
	TF-IDF
	TextRank
	主题模型
		LSA/lSI 奇异值分解
		LDA 吉布斯收敛
句法分析 PCFG
文本向量化
word2vec
	Bag Of Word 表示词向量的基础模型
	NNLM 预测下一个词
	C&W 打分
	CBOW skip-gram 上下文预测某词、某词预测上下文
doc2vec
	DW相比CBOW增加段向量 DBOW只输入段向量预测句子


构建知识图谱具体内容：https://www.jiqizhixin.com/articles/2017-09-14-4

知识抽取
	实体抽取（HMM CRF NN
	关系抽取（匹配触发词或依存分析树的pattern 
	半监督
		远程监督：从已有库中造句
		bootstrapping：抽取有种子实体的pattern
知识挖掘
知识融合
	计算属性距离
		levenshtein distance
		Wagner and Fisher Distance 增删改权重不同
	计算实体距离
		聚类 KNN
		Falcon-AO自动本体匹配系统
		Limes基于度量空间的实体匹配发现框架

知识推理
	根据旧关系推出新关系
	满足性(satisfiability) 本体是否有模型、概念是否是空集
	分类(classification) 母亲属于女人 女人属于人类 -> 母亲属于人类
	实例化(materialization) alice是母亲 母亲属于女人->alice是女人
							has_son(alice，bob) has_son属于has_child ->has_child(a,b)
	TBOX术语集，描述概念和关系的知识
	ABOX断言集，具体个人信息，一个对象是否属于一个概念，两个对象是否属于一个特定关系

	Tableaux算法检查可满足性
		FaCT++、Racer、Pellet、HermiT

