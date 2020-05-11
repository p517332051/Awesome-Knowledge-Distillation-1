# Awesome Knowledge-Distillation

- [Awesome Knowledge-Distillation](#awesome-knowledge-distillation)
  - [Efficient KD](#efficient-kd)
  - [Different forms of knowledge](#different-forms-of-knowledge)
    - [Knowledge from logits](#knowledge-from-logits)
    - [Knowledge from intermediate layers](#knowledge-from-intermediate-layers)
    - [Knowledge from Relations](#knowledge-from-relations)
    - [Graph-based](#graph-based)
    - [Mutual Information](#mutual-information)
    - [Self-KD](#self-kd)
  - [KD + GAN](#kd--gan)
  - [KD + Meta-learning](#kd--meta-learning)
  - [Data-free KD](#data-free-kd)
  - [KD + AutoML](#kd--automl)
  - [KD + RL](#kd--rl)
  - [Multi-teacher KD](#multi-teacher-kd)
  - [Knowledge Amalgamation](#knowledge-Amalgamation)
  - [Cross-modal KD & DA](#cross-modal-kd--da)
  - [Application of KD](#application-of-kd)
  - [Model Pruning or Quantization](#model-pruning-or-quantization)
  - [Other](#other)

## Efficient KD

1. Neural Networks Are More Productive Teachers Than Human Raters: Active Mixup for Data-Efficient Knowledge Distillation from a Blackbox Model. CVPR2020 (teacher是黑盒，假设有少部分unlabel数据，通过mixup扩充，用active learning从mixup得到的pool里选取student最不确定的那一部分放入training set，query teacher来获得这部分数据的soft label)

## Different forms of knowledge

### Knowledge from logits

1. Distilling the knowledge in a neural network. Hinton et al. arXiv:1503.02531 (KD，transfer logits)
2. Relational Knowledge Distillation.  Park, Wonpyo et al, CVPR 2019
3. Like What You Like: Knowledge Distill via Neuron Selectivity Transfer. Huang, Zehao and Wang, Naiyan. 2017
4. On the Efficacy of Knowledge Distillation. Cho, Jang Hyun and Hariharan, Bharath. arXiv:1910.01348. ICCV 2019
5. Improved Knowledge Distillation via Teacher Assistant: Bridging the Gap Between Student and Teacher. Mirzadeh et al. arXiv:1902.03393 (T和S容量差异过大时，加一个medium size的model过渡)
6. Self-training with Noisy Student improves ImageNet classification. Xie, Qizhe et al.(Google) CVPR 2020 (teacher训好后输入大量无标签图像，产生的output作伪标签，跟原始数据一起训student，给student的训练注入大量噪声，对robustness有很大帮助)
7. Knowledge Distillation via Route Constrained Optimization. Jin, Xiao et al. ICCV 2019 (用teacher的不同checkpoint监督student)

### Knowledge from intermediate layers

1. Fitnets: Hints for thin deep nets. Romero, Adriana et al. arXiv:1412.6550 (transfer feature)
2. Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer. Zagoruyko et al. ICLR 2017 (学attention)
3. A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning. Yim, Junho et al. CVPR 2017 (学中间层输入和输出的gram矩阵，paper所谓的解题思路)
4. Self-supervised knowledge distillation using singular value decomposition. Lee, Seung Hyun et al. ECCV 2018 (学SVD截断后的feature)
6. A Comprehensive Overhaul of Feature Distillation. Heo, Byeongho et al. ICCV 2019 (学pre-activation过了margin-relu之后的东西)
7. Knowledge distillation via adaptive instance normalization. Yang, Jing et al. arXiv:2003.04289 (学BN)

### Knowledge of Relations

1. Paraphrasing Complex Network:Network Compression via Factor Transfer. Kim, Jangho et al. NIPS 2018 (teacher端训个auto-encoder，得到的feature更容易被student学到)
2. Relational Knowledge Distillation.  Park, Wonpyo et al. CVPR 2019
3. Knowledge Distillation via Instance Relationship Graph. Liu, Yufan et al. CVPR 2019 (其实就是把一对一mimic和多对多mimic两种统一了一下)
5. Similarity-Preserving Knowledge Distillation. Tung, Frederick, and Mori Greg. ICCV 2019
6. Graph-based Knowledge Distillation by Multi-head Attention Network. Lee, Seunghyun and Song, Byung. Cheol arXiv:1907.02226
7. Correlation Congruence for Knowledge Distillation. Peng, Baoyun et al. ICCV 2019
8. Deep geometric knowledge distillation with graphs. arXiv:1911.03080

### Graph-based

1. Distillating Knowledge from Graph Convolutional Networks. Yang, Yiding et al. CVPR 2020 (GCN的KD，GCN中重要的是node之间的关系，所以KD就transfer这个关系，说白了就是每个node跟所有neighbor算距离然后做softmax，distill这个概率)

### Mutual Information

1. Contrastive Representation Distillation. Tian, Yonglong et al. ICLR 2020

### Self-KD

1. Learning Lightweight Lane Detection CNNs by Self Attention Distillation. Hou, Yuenan et al. ICCV 2019
2. `Rethinking Data Augmentation: Self-Supervision and Self-Distillation. Lee, Hankook et al. ICLR 2020 (被拒的，先学一个原始分类与SS变换分类，两种分类的笛卡尔积的大分类任务，再把这个模型distill给一个普通模型)`
3. Regularizing Class-wise Predictions via Self-knowledge Distillation. cvpr 2020 (用x同类的其他sample的预测结果来regularize网络关于x的logits，实质就是要求同类的所有图像之间的KL-div小，好处是：这是很好的regularization同时可以减小类内variance)

## KD + GAN

1. Distilling portable Generative Adversarial Networks for Image Translation. Chen, Hanting et al. AAAI 2020 (就是用KD训个小GAN)

## KD + Meta-learning

## Data-free KD
1. DAFL:Data-Free Learning of Student Networks. ICCV 2019 (出发点：试图生成原始数据分布，用GAN来做，但是因为缺少原始数据，所以无法训D，那么用训好的teacher做D，但teacher本身不是用来判别真假的，所以G的目标就是生成一些能让teacher产生某些output的图，对这些图有3个要求，一是teacher能给出one-hot的output，二是分类前的feature响应要尽量大，也就是L1大，三是类间尽量平均。有图之后，再用KD即可，但是不用ce-loss。G与student的训练交替进行)
2. Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion. Yin, Hongxu et al. CVPR 2020 (与7类似，其实data-free的paper核心都是如何生成数据，这篇用的DeepDream的升级版，deepdream本身是用label反推input，同时加上了TV loss和L2 loss来正则化input，这篇升级成了deep inversion，其实就是加了另外两个正则项，1是要求生成的图的bn统计量与原teacher相同，2是要求生成的图能增加student和teacher的JS散度，功能类似7中的generator，生成这些数据后再用常规的KD即可。生成挺慢的，140k ImageNet的图，需要8块V100跑一天。图相对逼真)
3. Data-Free Adversarial Distillation. Fang, Gongfan et al. CVPR 2020 (用GAN产生data来训student，整体是个minmax game，一方面对于产生的data，优化student使得与teacher的discrepancy减小，也即mimic，另一方面，优化generator使之产生增大discrepancy的data，student不断逼近老师，generator也不断生成还没有被学会的knowledge，比较有趣的一点是generator生成的image和label跟真实数据之间差异很大)
4. Knowledge Extraction with No Observable Data. Yoo, Jaemin et al. NIPS 2019 [[code][10.9]] (类似GAN结构，G的输入是y，产生x，x送入teacher应该再得到y，但只这样会导致多样性不足，同一个y产生一样的x，为增加diversity，G的输入还包括一个latent z，同时再增加一个decoder从x还原z，确保z与x的对应性，为进一步确保diversity，还要求一个batch产生的图尽量不同)

## KD + AutoML

## KD + RL

## Multi-teacher KD 

## Knowledge Amalgamation

## Cross-modal KD & DA

1. Multi-source Distilling Domain Adaptation. Zhao, Sicheng et al. arXiv:1911.11554 (跟KD没什么关系，1逐个source训分类器，2为每个source训一个encoder，将target的图映射到每一个source的feature space，通过WGAN来完成，训好的WGAN就有了计算样本间、分布间距离的能力，3从每个source domain里挑一半跟target domain接近的图finetune分类器，作者称这一步为distill个人觉得很牵强，4inference时把图过所有与source对应的encoder，再过每个source自己的分类器得到多个分类结果，这些结果加权，权重与每个source与target的W距离负相关。总结：每个source训分类器，WGAN去align target与每个source的feature从而可以使用每个source自己的分类器，align feature可以用一般的GAN来做，但是WGAN能有更好的距离度量，这为finetune source分类器和加权提供了基础)

## Application of KD

1. Structured Knowledge Distillation for _Semantic Segmentation_. Liu, Yifan et al. CVPR 2019 (semantic seg是一个结构性任务，所以针对性的KD就是pixel-pair similarity和holistic distillation也就是WGAN)
2. Teacher Supervises Students How to Learn From Partially Labeled Images for _Facial Landmark Detection_. Dong, Xuanyi and Yang, Yi. ICCV 2019 (与其说是teacher-student，不如说是actor-critic，并非将teacher的knowledge给student，只是teacher在给student的prediction打分。人脸关键点检测，有label数据和无label数据，用label数据训好若干student，再训练teacher能对这些prediction打分，groundtruth就是误差，之后student给出unlabel数据的prediction，teacher仍然可以打分，把分低的去掉，只保留好的，这些数据继续训student，循环往复)

## Model Pruning or Quantization

1. Knapsack Pruning with Inner Distillation. Aflalo, Yonathan et al. arXiv:2002.08258

## Other

1. Explaining Knowledge Distillation by Quantifying the Knowledge. [Zhang, Quanshi][18.13] et al. CVPR 2020

