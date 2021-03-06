# 神经网络在焊接过程中的应用现状及发展趋势分析

## 摘要

由于焊接过程是一个广泛使用但却极为复杂的工业过程，本文通过了解国内外神经网络在各个焊接过程中的应用情况，结合当前神经网络在机器学习以及深度学习中的发展情况，对现阶段神经网络在焊接中的研究及应用状况进行分析，并提出神经网络在焊接过程中的新的应用方向与发展方向。

鉴于焊接过程充满各种干扰信号，会极大地干扰传感信息的采集，采用拥有较高鲁棒性的神经网络对焊接过程进行预测或者控制会是比较好的选择。当前国内外已经有很多研究利用神经网络对焊接结构以及性能进行预测，且取得了不错的效果，但大多数使用的方式都是浅层的神经网络，少有使用深层神经网络对焊接过程进行预测，而近年来卷积神经网络已经在深度学习中取得了重大突破，因此使用卷积神经网络等深层神经网络对环境信息进行提取是焊接预测的一个发展趋势。

现阶段焊接的动力控制大多采用的是 PID 控制器，鲜有强化学习的应用，而强化学习也是近年来机器学习研究的一个热点方向，它非常适合与环境交互时不断进行学习，常用在机器人控制训练等连续决策任务中，因此强化学习将来也会在焊接领域得到广泛的应用。

关键词：机器学习; 焊接控制; 神经网络; 强化学习

## Abstract

Welding process is a widely used but complex industrial process. In this work, I understand the current situation of neural networks in various welding processes at home and abroad, combined with the current development of neural networks in machine learning and deep learning. The research and application status of the network in welding is analyzed, and the new application direction and development direction of the neural network in the welding process are proposed.

In view of the fact that the welding process is full of various interference signals, which will greatly interfere with the collection of sensing information, it is a better choice to predict or control the welding process using a neural network with high robustness. At present, there are many studies at home and abroad that use neural networks to predict welding structure and performance, and have achieved good results, but most of the methods used are shallow neural networks, and few use deep neural networks to predict the welding process In recent years, convolutional neural networks have made major breakthroughs in deep learning, so the use of deep neural networks such as convolutional neural networks to extract environmental information is a development trend in welding prediction.

At present, most of the power control of welding uses PID controllers, and there are few applications of reinforcement learning. Reinforcement learning is also a hot spot in machine learning research in recent years. It is very suitable for continuous learning when interacting with the environment. In continuous decision-making tasks such as training, reinforcement learning will also be widely used in the field of welding in the future.

Keywords: Machine Learning; Welding Control; Neural Networks; Reinforcement Learning

## 引言

焊接是一种以加热或加压方式对材料进行接合的工艺及技术。其操作要求严苛，对操作人员有着较高的熟练要求，且工作环境恶劣，操作人员常常处于高危且有毒的环境中，存在着很多的潜在危险[1]。而焊接机器人的应用能大大改善上述问题。它能够提供稳定且均一的焊缝，使得焊缝质量更有保障[2]。同时也大大改善了工业机械操作人员的生活和劳动条件，操作人员只需要方便地装卸工件，就已经可以有效避免操作中接触一些有毒的化学气体和弧光等有害元素。另外，由于焊接机器人可以二十四小时连续生产，因此它的生产效率更高。由此可见，智能化与自动化的生产已经逐渐成为了现代机械焊接加工技术进步与发展的必然趋势。

现代焊接机器人主要根据力矩、视觉、电弧等相关传感器对焊接环境的相关信息进行获取，经过智能化控制与调节来优化自身的焊接轨迹，从而完成复杂的焊接任务[3]。然而近年来焊接机器人的应用飞速增长，同时焊接机器人的应用领域也逐渐趋向复杂多变，焊接控制技术迎来了极大的挑战。由于焊接过程充满了强烈的弧光辐射、焊渣的飞溅、灰尘等不确定因素，还有着高温、氧化等等问题[4]，都会对焊接机器人所接收到的传感器数据产生影响，以至于焊接产生偏差，影响焊缝质量。为了使得焊接机器人能够应对这些问题，焊接机器人必须有比较强的抗干扰能力。

为了使得焊接机器人拥有智能控制能力，机器学习算法必不可少。机器学习算法以推理与学习为核心，涉及了概率论、统计学等多门学科，能够从复杂的数据中学习到知识并将其应用在实际生产中。在机器学习算法中最为突出的方法是神经网络模型，它拥有非常强的自适应能力以及极佳的抗干扰性，这意味它可以在焊接过程中拥有更高的控制精度以及更低的错误率。

随着近年来深度学习技术的发展，神经网络的性能不断提高，在图像处理等等很多领域都取得了突破性的进展，这为焊接机器人的智能化提供了强有力的技术支持。此外，神经网络还在强化学习等方面得到了广泛的应用，这使得焊接机器人能够应对更加复杂的环境，为焊接完全自动化控制奠定了基础。

## 1 神经网络研究现状

神经网络（Neural Network，NN）的相关研究在很早以前就已经出现，而如今已经形成了一个相当大的多学科交叉的学科领域[5]。通常所说的神经网络是指人工神经网络（ANN），是机器学习领域中一种常用的数学模型或计算模型，它在结构和功能上模仿了生物神经网络，有非常好的运算性能以及拟合能力。

### 1.1 神经网络基本模型

#### 1.1.1 M-P 神经元模型

生物神经网络最基本的单元是生物神经元（neuron）。生物神经元的基本结构如图 1.1 所示，当一个神经元的电位达到一个阈值时，它会转变为“兴奋”状态，进而向下一个神经元传递神经递质，以影响下一个神经元的电位。生物神经元彼此连接形成网状结构，从而能够完成复杂的功能。

![Figure 1.1](../_media/paper/Figure1.1.jpg)

图 1.1 生物神经元结构

1943 年，McCulloch 与 Pitts 将上述情形抽象为一个如图 1.2 所示的简单的数学模型，也就是人工神经网络中最经典的 “M-P 神经元模型”[6]。类似于生物神经元的概念，M-P 神经元接收前驱 $n$ 个神经元的输入信号，并对每个信号赋予一定的权重进行加和，得到 $\sum\limits_{i=1}^n w_i x_i$，将其作为该神经元的信号。当该信号值达到阈值 $\theta$ 时向下一个神经元传递信号，也就是向后一神经元传递信号 $y = f(\sum\limits_{i=1}^n w_i x_i - \theta)$。其中 $f$ 为激活函数，这里使用的是阶跃函数，也即 $f(x) = sgn(x)$。然而由于阶跃函数具有不连续、不光滑等问题，实际应用中常使用 $Sigmoid$ 函数作为激活函数。

![Figure 1.2](../_media/paper/Figure1.2.png)

图 1.2 M-P 神经元结构

#### 1.1.2 神经网络结构

类似于生物神经网络，将神经元彼此连接便形成了人工神经网络。单层神经网络只包含一个输入层和一个输出层，由于输入层神经元仅接收外界输入而不对数据进行处理，所以输入层往往不计入层数中。单层神经网络的处理能力有限，对于线性不可分的问题就束手无策了，所以常见的神经网络一般是两层以上的。两层的神经网络又称为单隐层神经网络（如图 1.3），它包含了一个输入层（input layer）、一个隐藏层（hidden layer）以及一个输出层（output layer）。输入层和输出层分别接收外界输入和将输出传输给外界，隐藏层和输出层都有着数据处理的功能。

![Figure 1.3](../_media/paper/Figure1.3.png)

图 1.3 单隐层神经网络结构

多层神经元相互堆叠形成的深层神经网络对数据有着极强的拟合能力，近年来神经网络层数的极限不断提高，在此基础上发展出来了新的学科——深度学习。深度神经网络的处理过程中，逐层将低层次的特征表示向高层次的特征表示转化，使得后续学习任务得到极大简化，只需要使用简单模型即可完成。近年来深度学习技术不断发展，在计算机视觉、自然语言处理等方面均取得了重大的突破。

#### 1.1.3 其他神经网络模型

M-P 神经元具有简洁且且高效的运算方式，这使得现在大多数神经网络都是基于 M-P 神经元的，深度神经网络也不例外。它们凭借着不同的网络结构，对某一种或一类特定的问题都有着较强的处理能力。比如擅长处理计算机视觉问题的卷积神经网络（Convolutional neural network, CNN）、擅长处理非欧拓扑结构数据的图神经网络（Graph neural network, GNN）、擅长自然语言处理的 Transformer 等等。

除此之外，还有一些神经网络采用了新型的神经元结构。比如神经网络中有一类模型是为网络状态定义一个“能量”（energy），网络的优化目标是将这个能量函数最小化。受限玻尔兹曼机（Restricted Boltzmann Machine，RBM）就是这样的一种网络，它常用对比散度算法来进行训练，能够取得不错的性能。近期张绍群、周志华等人参考了生物神经网络中树突浓度会在持续接受刺激后发生一定的变化，将该行为描述为一个二元二值的函数，并将该模型称为 Flexible Transmitter，简称 FT 神经元。FT 神经元具有时序记忆功能，使得其可以处理更加复杂的数据。并使用了复数运算模型巧妙地使神经元的两部分可以协同运算与更新。经测试，该模型在不同任务上均取得了不错的效果，在同等规模的前提下，甚至优于之前最优的神经网络[7]。

![Figure 1.4](../_media/paper/Figure1.4.png)

图 1.4 Flexible Transmitter 结构[7]

### 1.2 神经网络的优化方法

由于神经网络的输出结果 $\hat{y}$ 是与预期的输出结果 $y$ 有一定的误差的，这个误差的大小使用误差函数来描述，比如常用的均方误差（Mean Squared Error, MSE）就是计算实际输出与预期输出之间差值的平方和。为了使得神经网络的输出结果更加地精确，需要对神经网络的参数进行优化，常用的神经网络优化算法有误差逆传播算法、遗传算法、模拟退火算法等等。

#### 1.2.1 误差逆传播优化算法

误差逆传播（error BackPropagation，简称 BP）算法是如今最有效且最常用的神经网络优化算法[8]。BP 算法基于梯度下降（gradient descent）策略，向目标的负梯度方向调整参数，对于误差 $E_k$，在一次迭代后参数 $w_{hj}$ 的变化量为 $\Delta w_{hj} = -\eta \frac{\partial E_k}{\partial w_{hj}}$，其中 $\eta$ 为学习率。BP 算法根据误差对各个参数的梯度大小对参数分别调整，使得各个参数能够根据各自的“贡献”调整得恰到好处，提高算法的收敛速度与性能。

最基本的梯度下降法是对整个数据集遍历后计算梯度并进行更新的，这种方法也被称为批量梯度下降法（batch gradient descent, BGD）。由于每次更新都是针对整个数据集，所以不仅计算量大，而且会消耗大量的内存。另外，整个数据集不同数据之间的梯度存在一定的抵消现象，这就使得批量梯度下降法收敛非常慢。另外一种梯度下降法与批量梯度下降法恰恰相反，它每次对一对数据进行更新，这种方法称为随机梯度下降法（Stochastic gradient descent, SGD）。由于每遍历一个数据对就会计算梯度并更新参数，这使得随机梯度下降法对计算设备的性能要求更低，同时也使得网络收敛的更加快速。但由于每对数据都是有一定的随机性的，随机梯度下降的优化方向仅针对该次迭代所使用的数据对，所以随机梯度下降法更新过程中伴随着较大的噪声，更新的路线会较为曲折。为了综合上述两者的优点，小批量梯度下降法（Mini-batch gradient descent）每次采用一小批数据对网络进行更新，该方法不仅解决了批量梯度下降法收敛慢的问题，又解决了随机梯度下降法噪声多的问题，这也是神经网络更新中最常用的方法。

![Figure 1.5](../_media/paper/Figure1.5.jpg)

图 1.5 批量梯度下降、随机梯度下降、小批量梯度下降过程可视化

#### 1.2.2 启发式优化算法

BP 优化算法虽然能够非常有效地降低网络误差，但是它只是一种局部搜索算法，在训练过程中非常容易陷入局部最优的位置，导致进一步优化困难等问题。为了跳出局部极小以尽可能地逼近全局最小，人们常采用遗传算法（genetic algorithms, GA）、模拟退火（simulated annealing, SA）算法来逼近全局最优。

![Figure 1.6](../_media/paper/Figure1.6.jpg)

图 1.6 网络优化时的全局最小与局部极小

遗传算法（Genetic Algorithm, GA）是一种模拟达尔文生物进化论中的遗传学机理和自然选择的生物进化过程的计算模型，该方法是通过模拟自然进化过程对模型的最优解进行搜索的。它以个体（individual）为优化对象，以种群（population）为搜索空间，通过选择（selection）、变异（mutation）与交叉（crossover）选取出更优的个体。其本质是一种高效的搜索方法，并在搜索过程中自动获取和积累搜索空间的相关知识，对搜索过程进行自适应地控制，能够在全局范围内求得最优解。

<!-- 这里可以考虑是否详述 GA -->

模拟退火算法（simulated annealing, SA）是对冶金学退火过程进行模拟的算法。它受到热力学的理论的启发，将搜索空间内每一点想像成金属内的原子，将该位置的合适程度作为该原子的能量，而到达相邻位置的概率取决于两点“能量”之差。可以证明，模拟退火算法所得解依概率收敛到全局最优解。

此外，还有一些在 BP 优化的基础上进行优化的算法，比如 Momentum 优化等方法，上述方法都在一定程度上使得网络获得更好的性能，但由于这些技术大多都是启发式算法，理论上尚缺乏保障。

<!-- 这里可以考虑是否详述 Momentum -->

#### 1.2.3 神经网络的优化难点及解决方案

正由于神经网络具有强大的表示能力，神经网络经常遭遇过拟合问题。为了解决该问题，需要使用正则化（regularization）或 Early Stopping 等方案。

在网络训练初期，训练误差与验证误差往往是同时下降的，此时网络学到了比较泛化的表示。随着训练迭代次数的增加，网络可能会将训练集中独有的特征当作所有潜在样本所具有的一般特征，这将会降低网络的泛化能力，该现象被称为“过拟合”。这在误差上表现为训练误差仍在下降而验证误差不下降或反而上升。为了使得模型学习到最泛化的表示，可以在刚刚发生过拟合时停止网络的训练，这便是 Early Stopping。

另外，网络能够过拟合的主要原因是网络具有极高的拟合能力，如果能够限制网络的拟合能力，那么网络将会更倾向于学习泛化的表示。该方式被称为正则化，常用的正则化方法有 L2 正则化与 dropout。L2 正则化方式通过降低权重的 L2 范数以达到限制模型的拟合能力，提高模型的泛化能力。dropout 在训练过程中随机使某些神经元失活的情况下依然要求神经网络输出预期结果，使得整个网络不过分依赖于任一个神经元，从而减弱过拟合现象[9]。

梯度弥散（vashing gradient）问题是一种在深度神经网络中常见的问题，当使用反向传播方法计算导数的时候，随着网络的深度的增加，反向传播的梯度的梯度值会急剧地减小。由于 BP 神经网络早期使用的激活函数是具有饱和特性的 Sigmoid，其在较小或者较大的输入时导数趋于 0，也就导致了梯度无法传递，参数不能得到更新。

在深度学习崛起的前夕，有两个深度模型突破了限制，达到了更深的网络结构，一个是深度信念网络（deep belief network，DBN）[10]，一个是堆叠自编码器（stacked auto encoder）。两者有着相似的结构与训练方式，前者是由多个受限玻尔兹曼机（Restricted Boltzmann Machine，RBM）相互堆叠后添加一个分类器而成，后者是由多个自编码器（auto encoder，AE）堆叠而成，它们的训练方式都是先进行逐层无监督预训练（pre-training），后使用有监督的微调（fine-tuning）。这种训练方式的有效性主要来源于无监督预训练可以降低网络的方差，起到了一定的正则化效果。另一方面，随机梯度下降的网络训练的早期行为会造成很大的影响，这可能导致后期无论多少次迭代都无法跳出某一个局部最小，而无监督预训练能够将网络参数初始化在一个较好的位置，使其更加接近全局最优[11]。然而该方法并没有解决梯度弥散的问题，所以神经网络的深度仍然非常受限。但该方法确实可以在一定程度上提高模型准确度，而且在拥有大量无标签数据与少量有标签数据的情况下，该方法是更为合适的选择，这种训练方式也被称为半监督学习（Semi-supervised learning）。

![Figure 1.7](../_media/paper/Figure1.7.png)

图 1.7 堆叠自编码器结构

为了解决梯度弥散的问题，Xavier Glorot 提出对参数进行合理的初始化以保证前向传播与反向传播时数据的分布不会发生太大改变[12]，随后 Geoffrey Hinton 提出使用 ReLU 作为激活函数[13]，这也是如今最常用的激活函数。但 ReLU 激活函数在负值时仍然没有梯度，会导致神经元死掉，Andrew Maas 提出了 LeakyReLU 激活函数进一步缓和了该问题[14]。但网络更深的时候，梯度弥散就会死灰复燃，何恺明等人在 ResNet 提出了残差训练的概念，极大地提高了神经网络的深度[15]。

### 1.3 深度神经网络研究及其应用现状

由于神经网络在结构上非常容易堆叠成多层结构，而神经网络的每一层非线性变换都会使得模型的的复杂程度提升，这就使得深层网络有着明显更高的表示能力。因此，提高神经网络深度的极限是近年来神经网络研究的一个热门方向，由此也演变出来了深度学习这一门新的学科。深度神经网络非常适合高维结构化数据的处理，它使得人们不必纠结于高维数据的繁杂处理步骤，只需要定义输入与输出即可获得一个准确的预测模型，神经网络便会自行选用合适的方式对数据进行处理。现如今诸如计算机视觉、自然语言处理等等很多复杂的学习任务都交由深度神经网络来处理，且均取得了重大的突破。

#### 1.3.1 神经网络在计算机视觉的研究与应用现状

神经网络在很早以前就已经应用于计算机视觉任务中了，人们最初尝试的是传统的神经网络，但由于传统神经网络模型并没考虑到图片各个像素点之间的空间关联性，这就使得神经网络拥有过多的参数，同时非常难以训练，很难应用于实际生产中。后来卷积神经网络（Convolutional Neural Network, CNN）的发明与应用彻底改善了这一问题，神经网络不仅在计算机视觉中广泛应用，甚至达到了超越人类水准的分类辨识能力。

1980 年，Kunihiko Fukishima 提出的 Neocognitron[16] 创造性地从人类视觉系统引入了许多新的思想到人工神经网络，被广泛认为是 CNN 的雏形。1990 年，LeCun 将反向传播应用到了类似 Neocoginitro 的网络上来做监督学习[17]。LeCun 的网络结构中引入了权值共享的概念，空间上不同位置的神经元使用同一个卷积核进行卷积。权值共享大大地减少了参数量，也就降低了过拟合的风险，提高了模型的泛化能力，另外也使得训练的速度大大提升。1998 年，LeCun 提出的 LeNet-5[18] 技压群雄，轻松超过其它主流的机器学习算法。但由于当年计算能力的限制和后来 SVM 的大放异彩，CNN 在 21 世纪初迷失了近十年。

![Figure 1.8](../_media/paper/Figure1.8.png)

图 1.8 LeNet-5 网络结构[18]

LeNet-5 的网络结构已经与现在人们常用的卷积神经网络结构非常相似，它主要主要包含卷积层与池化层。卷积层是对图片上局部像素值进行加权求和，建立一个局部感受野，降低网络参数量。如图 1.9 所示，卷积核扫描过图片时对图片相应位置进行加权求和，将所得值作为输出“图片”相应位置的值，之后通过卷积核不断移动，扫描得到完整的输出“图片”。输出的“图片”称为特征映射（feature map），经过卷积后的 feature map 表示了更复杂的特征[19]。而池化层是对图片进行下采样，降低图片的大小，从而在减少数据量的同时尽可能地保留有用的信息[5]。

![Figure 1.9](../_media/paper/Figure1.9.jpg)

图 1.9 卷积层处理过程

随着计算机硬件的快速发展，硬件的计算能力不断提升，神经网络的训练逐渐成为了可能。2006 年，研究人员成功利用 GPU 加速了 CNN，相比 CPU 实现快了近四倍。2012 年，AlexNet[20] 在 ImageNet 大规模识别挑战赛（ImageNet Large Scale Visual Recognition Competition，ILSVRC）图片分类任务上以 15.3% 的 Top-5 错误率登顶，远高于第二名的 26.2%。AlexNet 基本结构参考了 LeNet-5，同时为了防止梯度弥散（vashing gradient）等问题使用了 ReLU 激活函数[9]。AlexNet 使得 CNN 再度被人们所重视，也标志着神经网络的复苏与深度学习的崛起。

随后几年，CNN 迎来了快速发展的浪潮，新的网络结构层出不穷，效果也在不断提升。2014 年，Google 提出的 GoogleNet[21] 和 Visual Geometry Group 提出的 Vgg[22] 分别在 ILSVRC2014 位列第一和第二。后者在 AlexNet 的基础上进一步提高网络深度，前者则在网络结构上另辟蹊径，不仅能够提高网络深度，而且大大减少网络参数量。2015 年，何恺明等人提出的 ResNet 利用残差结构使得网络能在不退化的前提下提升到 152 层[15]，一举摘得 ILSVRC2015 桂冠。此后几年，Google 在 GoogLeNet 提出的 Inception 结构和何恺明等人提出的 ResNet 成为了 CNN 两个主要发展的方向。Google 在 Inception 结构的基础上提出了 InceptionV2[23]、InceptionV3[24]、Xception[25]、InceptionV4[26]、Inception-ResNet[26] 等结构，不断提升网络的性能，最新的 Inception-ResNet 融合了 ResNet 的残差结构，使得网络性能进一步提升。何恺明等人也相继提出了 ResNetV2[27]、ResNeXt[28]，后者也在结构中借鉴了 Inception 结构。2017 年，GaoHuang 提出的 DenseNet 结构建立了比 ResNet 更加密集的连接，并提出了特征重用的概念，不仅能够解决梯度弥散等问题，还进一步减少了参数量[29]。同年，Jie Hu 提出的 SE-Net 利用通道注意力机制进一步优化 ResNeXt 结构[30]，一举夺得 ILSVRC2017 同时也是最后一届 ILSVRC 的桂冠。

![Figure 1.10](../_media/paper/Figure1.10.png)

图 1.10 ResNet 中的残差块结构[15]

最近两年的卷积神经网络结构研究一方面在逐渐倾向于轻量化，以适应移动设备以及嵌入性设备，典型的神经网络有 MobileNet[31] 系列。另一方面，卷积神经网络在结构的设计中尝试使用神经网络来设计，比如 Google Brain 的 NASNet 采用了强化学习来对神经网络结构进行搜索[32]。另外，Google 的 EfficientNet 研究表明卷积神经网络的深度、宽度以及分辨率有着较高的相关性，想要设计较好的神经网络结构，需要对这三个参数进行准确的调控[33]。何恺明等人最近提出的 RegNet 并不专注于设计单个网络实例，而是对网络设计空间进行整体估计，获得最优深度神经结构[34]。

![Figure 1.11](../_media/paper/Figure1.11.png)

图 1.11 RegNet 中对网络设计空间的搜索过程[34]

神经网络的研究极大地推动了计算机视觉的发展，它使得原来很多不可能的任务成为了现实，如今神经网络已经在很多任务上达到甚至超越人类的表现，这让我们看到了深度学习研究的无限可能。

#### 1.3.2 深度强化学习研究与应用现状

强化学习（Reinforcement learning, RL）是机器学习和人工智能的一个分支，专注于目标导向的学习和决策。强化学习不同于传统有监督的机器学习任务，在最开始是没有可供训练的数据的，而数据的获取是通过机器与环境之间的交互逐渐获得的。在与问题或环境的持续互动中，强化学习智能体采取行动并观察所得的奖励，智能体根据这些观察结果来改变其选择动作的方式，以达到学习的效果。简而言之，监督学习就是教书式的学习，而强化学习则是在实践中进行摸索式的学习。

强化学习主要分为基于模型（model-based）、基于值（value-based）、基于策略（policy-based）三种学习方式。基于模型的学习方法是让智能体自己学习出来一个能够从它的观察角度描述环境是如何工作的模型，然后利用这个模型来进行动作的规划。也就是说基于模型的方法需要对环境进行建模，建模的好坏则直接影响了最终决策的优劣。然而事实上很多情况并不需要对环境进行建模就可以找到最优的策略，基于值的学习方式与基于策略的学习方式就是最好的证明。基于值的学习方式是让智能体能够更加准确地进行自我评价，智能体可以在采取行动之前就意识到自己改行动的好坏，进而及时调整行为。基于策略的学习方式则是直接让智能体学习在某一环境下自己应当采取何种行为。这三种方式并非完全割裂的，事实上模型、值与策略刚好是强化学习智能体的三个重要组成部分，而三种不同的学习方式只是学习的关注点不同而已。

![Figure 1.12](../_media/paper/Figure1.12.png)

图 1.12 强化学习系统结构图

最近几年强化学习的常用方法也在逐渐转变，过去常用的方法是基于模型的学习方法，而近期的研究中已经很少使用该方法，更多使用的是基于值与基于策略的学习方法。自 12 年 Actor-Critic 算法[35]被提出，研究的热点也转向了基于值与基于策略相结合的方法。此后，2013 DeepMind 提出 A3C 算法，通过多线程异步训练的方法解决了数据时间相关性的问题，使得 Actor-Critic 网络训练更加容易[36]。

![Figure 1.13](../_media/paper/Figure1.13.png)

图 1.13 Actor-Critic 算法流程

近年来深度学习的崛起也是极大地推动了强化学习的研究，强化学习的研究重心也在逐渐向深度强化学习转变。13 年 Volodymyr Mnih 等人提出的 Deep Q-Network 成功地将深度学习方法应用在基于值的学习方法 Q-learning 上，被人们认为是深度强化学习的开山鼻祖[37]。15 年到 16 年 DeepMind 推出的 AlphaGo 先后击败了欧洲围棋冠军华裔法籍职业棋手樊麾二段与世界冠军韩国职业棋手李世石九段，这使得人们真正地意识到人工智能的能力，AlphaGo 也被誉为人工智能研究的标志性进展。AlphaGo 采用蒙特卡洛树搜索与深度神经网络相结合，其中神经网络的训练一方面使用了人类棋谱来进行监督训练，另一方面采用了基于模型、基于值与基于策略三种方法相结合的强化学习方法[38]。此后，DeepMind 再次推出 AlphaGo Zero，抛弃从人类棋谱中进行监督学习的方法，而是从零开始，不断进行自我对弈，这使得学习速度大大提升，仅仅 40 天就超过之前所有版本的水平[39]。

深度强化学习最近几年在不同领域大显神通，比如在视频游戏、棋类游戏上打败人类顶尖高手，控制复杂的机械进行操作，调配网络资源，为数据中心大幅节能，甚至对机器学习算法自动调参。各大高校和企业纷纷参与其中，提出了眼花缭乱的 DRL 算法和应用。DeepMind 负责 AlphaGo 项目的研究员 David Silver 甚至喊出认为结合了 DL 的表示能力与 RL 的推理能力的 DRL 将会是人工智能的终极答案。然而强化学习仍然是具有非常多的局限性的，比如样本利用率非常低，好的奖励函数难以设计，难以平衡“探索”和“利用”，对环境的过拟合等等问题，这些问题严重地阻碍了深度强化学习的发展，因此这些问题将会是深度强化学习今后的研究重点。

## 2 神经网络在焊接领域的应用研究现状

智能化是当今焊接领域的重要发展方向，而如今人工智能也是正值发展的上升期，各种模型层出不穷，在很多任务上都取得了重大的突破。焊接领域的很多任务也都开始应用新的智能技术，由于神经网络是一种非常有效的机器学习算法，它拥有非常强大的学习能力，可以轻松从数据集中发现潜在规律，将神经网络应用于焊接领域中的焊接工艺参数选择与优化、焊缝跟踪、焊接缺陷预测、力学性能预测等都具有比较理想的效果[40]。

### 2.1 机器学习算法在焊接领域的研究现状

机器学习是人工智能的一个分支，涉及统计学、逼近论、概率论、凸分析、计算复杂性理论等多门学科。机器学习的目标是让计算机在设定的任务上拥有自主学习能力，它是一类从数据中自动分析获得规律，并利用规律对未知数据进行预测的算法。机器学习主要关注的是行之有效的学习算法，由于很多问题难以使用硬编码程序来解决，所以部分的机器学习研究是开发容易处理的近似算法。机器学习中的算法繁多，常用的算法主要是决策树（decision tree）、支持向量机、神经网络、k 近邻（k-nearest neighbor，k-NN）算法，这些算法现如今已经广泛用于焊接预测与控制等应用之中，为焊接过程提供了强有力的技术保证。

![Figure 2.1](../_media/paper/Figure2.1.png)

图 2.1 机器学习的基本流程

支持向量机可以在高维空间中构造超平面以最大程度地将两类数据分离，支持向量机会最大化两类数据相对于这个超平面之间的距离，这个距离被称为间隔（margin），因此这种线性分类器也常被成为最大间隔分类器[41]。然而这种线性分类器无法对非线性数据进行划分，所以需要在其基础上应用一个核函数，使 SVM 具有非线性分类的能力。由于 SVM 相比于 logitic 回归拥有更加简单的数学表示形式，所以 SVM 在小型的分类任务上往往会有着更好的效果和训练速度。顾盛勇在焊缝视觉检测系统中使用了 SVM 预测焊缝位置，并利用 PCA 算法进行优化[42]。贺峰等人基于多传感器和 SVM 对焊接过程进行模式识别，有效地提高了焊接过程准确率[43]。

随机森林算法也是一种机器学习任务中的常用方法。随机森林算法是一个包含多个决策树的分类器，并且其输出的类别是由个别树输出的类别的众数而定。它对多种数据都能较好地进行分类，但是在类别特别多时它的表现并不会太好。Zhifen Zhang 等人首先使用 PCA 算法对数据进行降维处理，之后使用 RF 进行分类，有效地对焊缝缺陷种类进行预测[44]。Haixing Zhu 等人首先利用 CNN 对熔池图片进行降维，之后使用 RF 进行了有效的分类[45]。

### 2.2 神经网络在焊接结构及性能预测中的应用现状

焊接接头的结构能够直接影响焊接件的性能，间接地反映了焊接质量，因此如果能够根据焊接时所使用的工艺参数来对焊接熔宽进行预测，那么就能实时判断焊接质量，进而调整焊接参数以达到焊接过程的监控。但由于焊接过程涉及多种复杂因素，很难人工推导出对焊接结果的预测公式，因此常需要机器学习技术对已有数据进行拟合，获得较为准确的预测模型。其中最有效的方法是神经网络，它在非线性、多变量复杂的预测问题都取得了比较好的效果[46]。

#### 2.2.1 神经网络焊接接头结构分析研究

传统的 BP 神经网络在焊接预测中已经有着广泛的应用，均取得了不错的效果。赵成涛建立了基于 BP 神经网络的镁合金焊缝成形预测模型，利用神经网络的映射能力和分析能力，采用焊接过程的焊接速度、焊接电流、焊接电压、焊丝干伸长作为预测输入，把焊缝成形中的焊缝熔深、熔宽、余高作为信息输出对神经网络进行训练，从而建立基于 BP 神经网络的焊接参数和焊缝成形的映射模型[47]。黄军芬等人利用 BP 神经网络建立了熔池形貌与背面熔宽的模型，对熔透情况进行了预测[48]。Limeng Yin 等人利用双隐层神经网络对焊缝尺寸进行了预测，在网络的设计细节中使用了较新的深度学习技术，取得了较高的预测精度[49]。

虽然传统的 BP 神经网络已经能够取得不错的效果，但由于其本质是一种局部搜索方法，很容易陷入局部最优，使得网络泛化能力下降。遗传算法受到自然界生物进化机制的启发，借鉴了孟德尔的遗传学说和达尔文的进化论，能够在全局范围内对模型参数进行搜索和优化。Bowen Liu 搭建神经网络来拟合焊接参数与焊接缺陷的映射关系，并使用遗传算法进行优化，并进一步搜索焊接缺陷最低时对应的焊接参数，最终焊接成品有着较好的质量[50]。

遗传算法不仅可以独立使用，还可以结合 BP 算法进一步提高模型的精度，其方法是首先利用遗传算法优化神经网络参数达到一个较优的效果，然后以此为 BP 优化的起点，使用梯度下降法进一步训练以降低误差。该方法被称为 BP-GA 方法，它结合了 GA 和 BP 的优点，首先在全局范围内为 BP 设置了一个比较好的起点，使得之后的 BP 优化能够更加接近全局最优。张喆等人利用 BP-GA 的方法拟合了焊接前进方向温度预测模型和前进侧方向温度预测模型，其预测能力要优于基于传统 BP 神经网络所建立的模型[51]。Hua Ding 等人利用 BP-GA 算法对焊接残余厚度进行了预测，使得网络的预测精度更加准确，收敛速度大大提高[52]。

![Figure 2.2](../_media/paper/Figure2.2.jpg)

图 2.2 BP-GA 神经网络权值阈值优化流程[51]

此外，无监督预训练也能够显著提高神经网络的性能。稀疏自编码器（sparse auto encoder，SAE）是自编码器的一种变体，它对参数添加了一个惩罚项，使得被激活的神经元限制数量限制在一个较低的值，换言之，网络会学习到较为稀疏的表示，而稀疏的表示能够在编码时更加有效地对数据进行降维。堆叠稀疏自编码器（stacked sparse auto encoder，SSAE）是由稀疏自编码器堆叠而成的，它相比于堆叠自编码器拥有更好的降维效果，可以学习到更有用的特征。Yanxi Zhang 等人利用 SSAE 对焊接状态进行预测，之后使用 GA 进行超参数搜索，取得了一定的效果，并将该方法称为 SSAE-GA 方法[53]。

![Figure 2.3](../_media/paper/Figure2.3.png)

图 2.3 SSAE-GA 对神经网络的优化流程[53]

与 SAE 相似，还有一种相似的自编码器被称为降噪自编码器（denoising auto encoder），堆叠后的结构即 SDAE，SDAE 能够从包含噪声的数据中还原数据，因此有着更强的鲁棒性。Johannes Gunther 等人在图像特征提取过程中使用了 SDAE，并使用提取出的特征进行了 SVM 分类测试，仅产生了较低的分类误差。这些提取出的特征用于后续焊接控制神经网络的输入，取得了较好的控制效果[54]。

#### 2.2.2 卷积神经网络熔池特征提取研究

焊接接头的性能直接受焊缝质量影响，而焊缝质量又直接受焊接时熔池影响，因此熔池形貌信息能够辅助焊接接头结构预测以及焊接性能预测。由于熔池信息往往只能通过图片采集的方式获得，而从图片中获得重要的熔池形貌信息往往需要图片处理技术，常用的熔池形貌提取技术有阈值分割、边缘检测算子、数学形态学方式。由于这些方法往往效果并不是很好，而且处理过程较为繁琐，因此该任务可以采用神经网络来完成。

图像处理所使用的神经网络主要是卷积神经网络，卷积神经网络中的特征提取和模式分类同时进行，对图像的位移、缩放、倾斜及其它形式的扭曲变形都具有良好的不变性和等变性[55]。而且卷积神经网络有着很强的学习能力与自适应能力，相对于传统的边缘提取技术，它拥有更高的鲁棒性与更快的处理速度。

深度学习的多层堆叠、每层对上一层的处理机制可以看作对输入信号进行逐层加工，从而把初始的输入表示一步步转化为与输入目标更为密切的表示，由此可将深度学习理解为特征学习[5]。如图 2.4 所示，卷积神经网络在较低的层次会处理一些简单的诸如边缘、形状、颜色之类的特征，而在高层，低层特征组合出来更为复杂的对象特征，诸如眼睛、嘴、耳朵等等[19]。因此即便输出结果有所不同，神经网络在前几层也会做几乎相同的特征提取任务。

![Figure 2.4](../_media/paper/Figure2.4.png)

图 2.4 可视化卷积神经网络各层所提取的特征[19]

近几年有很多利用卷积神经网络进行熔透预测的相关研究，其本质上也是低层网络从熔池图像中提取出有用的特征，交由高层预测模块进行预测。李海超等人使用卷积神经网络对熔透状态进行了预测，并对卷积层的特征映射进行了可视化，对熔透预测模型的依据进行了一定的解释。刘新锋利用基于 ImageNet 预训练的 AlexNet，对穿孔状态及熔透状态进行预测，取得了较高的准确率[56]。

也有一些研究使用卷积神经网络直接将熔池图像特征进行压缩，提取出较低维度的特征信息，作为后续预测步骤的一部分信息。覃科等人利用卷积神经网络对熔池特征进行提取，交由后续支持向量机（support vector machine，SVM）分类器进行分类预测，两者同步训练，以预测焊接熔池的状态信息。他们同时还使用了传统的 BP 神经网络直接进行预测，对比结果显示卷积神经网络明显有着更高的准确率和鲁棒性[57]。Haixing Zhu 等人利用卷积神经网络对焊缝表面图像进行特征提取，交由后续随机森林（Random Forest，RF）算法进行分类，优于直接在 CNN 后接 Softmax 分类层的效果[45]。

![Figure 2.5](../_media/paper/Figure2.5.png)

图 2.5 熔池状态识别的卷积神经网络结构[57]

#### 2.2.3 卷积神经网络焊接缺陷识别研究

焊接缺陷极大地影响焊接的质量，焊接过程应尽可能避免缺陷的产生，目前大多数焊接缺陷检测技术仍然采用传统的人工检测图像法，这需要质检人员通过主观经验来对焊缝中所存在的缺陷进行分类与定位。而随着焊接作业效率的不断提高，人工在线检测将会成为整个作业生产线的效率瓶颈，而且人工检测主观影响因素较大，检测的结果可能可靠性比较低[58]。

近年来很多研究人员提出了基于神经网络的焊接缺陷检测方法，但大多数仍然需要先使用图像处理从图像中对主要特征进行提取，然后使用浅层神经网络来对缺陷进行识别与定位。而最新的计算机视觉技术使用的卷积神经网络有着强大的特征提取能力，可以极大地精简图像预处理技术，并取得更好的效果。杨志超等人直接将预处理后的焊缝 RT 图像作为输入，传入卷积神经网络进行处理，最终输出焊缝的缺陷种类，实验结果的识别率与泛化能力都非常良好，能够对焊缝缺陷种类以及质量进行良好的划分[59]。刘梦溪等人采用深度置信网络来尝试对焊缝缺陷进行分类识别，相比于传统 BP 神经网络以及 SVM 有着更高的精度，而且当网络深度越深时，网络效果往往越好[60]。此外，刘梦溪等人还尝试采用卷积神经网络来对 X 射线焊缝来进行识别，基于 CUDA-CONVNET 提出了一种新的 CNN 结构，缺陷识别率达到了 93.26%[61]。Hou 等人采用了卷积神经网络从 X 射线图像中提取高层次特征，在训练数据的采集过程中使用了三种重采样方法，最终识别准确率达到了 97.2%[62]。黄旭丰采用生成对抗网络对焊接缺陷样本进行数据增强，使用 MobileNet 网络对焊接缺陷进行分类，并采用迁移学习策略提高模型的收敛速度与泛化能力，使用 YOLO 算法对焊缝缺陷位置进行定位，取得了非常高的检测效率与精度[63]。

#### 2.2.4 神经网络焊接性能预测研究

焊接的性能直接反映了焊接的质量，如果能根据焊接时的焊接工艺参数等焊接实时信息对焊接性能进行预测，能够更有针对性地提出优化焊接过程参数以及相应的控制方法，也就达到了焊接的在线监控的效果。张永志等人利用广义动态模糊神经网络来对不同厚度、不同工艺条件下的 TC4 钛合金 TIG 抗拉强度、抗弯强度、焊缝硬度、热影响区等力学性能进行较为准确的预测[64]。刘政军等人提出利用双重人工神经网络来对铝合金焊接强度系数进行预测，该模型能够较好地对焊接工艺参数、接头力学性能与显微组织结构之间的关系进行拟合，最终计算结果与物理测试数据基本一致[65]。刘立鹏等人利用 BP-GA 算法优化神经网络，对焊接接头力学性能进行预测，并达到了预期的精度要求[66]。张昭等人在搅拌摩擦焊过程中使用 BP 神经网络成功地通过搅拌头旋转速度、焊接速度、距离焊接中心距离等参数预测了焊接接头硬度，且测试结果表明 BP 人工神经网络能够很好地预测接头硬度，为焊接接头力学性能预测提供了新方法[67]。

另一方面，如果通过焊接后测量的焊缝尺寸参数来预测焊缝性能虽然无法直接对焊接参数进行调整，但它可以作为一个预测模块来使用，在前驱预测模块通过焊接工艺参数预测得到焊接接头结构时，进一步通过该模块获得焊接性能，以辅助更加精确地反向参数调控。同时，这也对焊接性能研究起到了重要的作用。阮德重等人考虑到时效强化后焊接接头的抗拉强度与焊缝的形状密切相关[68]，尝试通过焊缝形状参数来预测和控制接头的力学性能，使用了 RBF 神经网络对接头焊缝形状与其对应的焊接接头的抗拉强度等参数进行预测，精度大幅提高[69]。

### 2.3 神经网络在焊接过程控制中的应用现状

焊接的工艺控制目标是获得较好的焊缝质量，一方面可以通过焊接接头的力学性能能够直接反映焊缝质量，另一方面可以通过焊缝的尺寸来间接反映焊缝质量，而这两者我们可以通过焊缝结构分析与焊缝性能预测获得。因此，一种简单的控制方式就是，在焊接机器人的工作过程中引入实时地预测焊接质量，并搜索到最佳质量所对应的焊接参数，并进行焊接。这一方面已经有很多相关研究，而且前面所述的结构分析与性能预测算法也可以直接应用。张抱日等人利用 BP 神经网络建立了焊接高度与电弧电压的对应关系模型，并进一步实现了焊接高度的自动控制[70]。

焊接的完全自动化需要焊接机器人自行跟踪焊缝并进行精准地焊接，而在这个过程中，焊接偏差的大小也会直接决定着焊接质量的高低。由于焊接是一个高度自由的动态过程，机器人动力学结构上高度相关与耦合，难以直接建立精确的机械臂动力学模型，另一方面焊接过程存在大量的不确定的扰动，这些都会对焊接控制产生不可忽略的误差，因此焊接过程的动力学控制也是一个重要的研究课题。

#### 2.3.1 神经网络在焊缝跟踪过程的应用

焊接的动力学控制的关键是使得机器人能够对焊缝进行跟踪，这需要通过传感器获得视觉等信号来引导机器人姿势以自动调整焊缝的位置，然后从起始点开始对焊缝进行自动跟踪的控制。其中所使用的传感信号包含了焊接过程中的坡口与焊炬的图像信息以及焊接过程所产生的光、热、电、磁、声等物理信号。这些信号通过图像处理技术以及控制算法的处理得到焊缝的中心位置以及焊枪的调整方向。

![Figure 2.6](../_media/paper/Figure2.6.png)

图 2.6 焊接机器人焊缝自动跟踪系统框图

由于焊接的偏差会极大地影响焊缝质量，而焊接过程中影响因素众多，传统的 PID 控制很难实现焊缝跟踪的自动化控制，因此焊缝跟踪过程也需要神经网络等机器学习技术的辅助。陈皓等人使用神经网络来建立焊缝跟踪多目标优化模型，并通过遗传算法对神经网络进行优化，并与 BP 优化进行对比，结构表明该方法具有更高的焊接精度以及更快的焊接速度，有效地解决了焊接机器人中姿态的调整问题以及路径的规划问题[71]。刘建昌等人针对机械臂轨迹控制问题，建立了动力学方程，并通过神经网络来对跟踪误差进行补偿，提高了控制策略的自适应能力和泛化能力，进而提高了机械臂的控制精度[72]。张文辉等人在机器人非线性控制的过程中采用了神经网络来补偿不确定模型，并通过变结构控制器来消除逼近误差，两者相互结合、相互促进，使得系统能够保持强鲁棒性，提高了系统的全局稳定性[73]。王保民等人尝试对机械臂末端直接控制，提出了笛卡尔坐标系下的轨迹跟踪控制算法，并通过 RBF 神经网络来对机械臂模型误差进行补偿，提高了控制算法的精度[74]。

![Figure 2.7](../_media/paper/Figure2.7.png)

图 2.7 RBF 神经网络结构[74]

#### 2.3.2 深度强化学习在焊接控制中的应用

现阶段传统的机器人控制算法大多使用的是传统的路径规划算法，但这些算法只能应用在特定的某些场景，缺乏灵活性，而且需要对环境进行精确的建模，在复杂的工业环境下并不能很好地适用。强化学习能够在与环境的交互过程中进行学习，已经在机器人的控制过程中得到应用广泛。因此，如果将强化学习应用在焊接机器人的控制过程中将会取得更佳的控制效果，降低算法的复杂度与错误率。

Q-learning 是一种常见的基于值的强化学习算法，但该算法需要将所有状态、行为以及对应的值列在一张表上，而焊接环境对应着无数的状态与行为，不仅难以枚举，而且会占用大量的内存空间。DQN 是 Q-learning 的改进版本，它使用神经网络替代了 Q-learning 中的表格，使得该算法在复杂的环境中也可以使用。李广创等人在 DQN 中使用三层的神经网络，将机械臂作为输入的状态信息，机械臂的运动关节角度作为输出的控制信号，通过离线训练使得机械臂能够自行找到一条接近于最优的运动轨迹，能够成功地避开障碍物到达目标点[75]。

Q-learning 在网络更新的时候采用了 off-policy 的策略，它在计算下一状态的预期收益时选取最优的行为，而当前策略与学习时的策略不同，因此实际并不一定采取最优的行为。为了改进该问题，Sarsa（State-Action-Reward-State-Action）采用 on-policy 策略对网络进行更新，在当前的 policy 直接执行一次行为的选择，然后用这个样本来更新策略，因此生成的样本的策略和学习时的策略相同，该方法直接了当，更容易使用。刘卫朋将 PD 控制器与 Sarsa 控制策略相结合，并将其用于焊接两自由度的机械臂控制问题，其中 PD 用于基本的控制任务，而 Sarsa 算法则用于对未知干扰因素进行补偿，提高机器的泛化能力。而且通过仿真实验可以的值，在引入强化学习算法后，机械臂不仅可以实现自主控制，而且明显提高了学习速度[76]。

![Figure 2.8](../_media/paper/Figure2.8.png)

图 2.8 基于 SARSA 的机械臂轨迹跟踪控制系统方案[76]

Actor-Critic 算法是一种基于策略与基于值训练方式相结合的强化学习算法，它采用一个 Actor 来学习用来生成行为的策略，以及一个 Critic 来评估算法的性能，以降低对学习算法更新时的方差。Actor 与 Critic 的协同工作使得算法能够有效的进行，同时两者相互促进，使得有着更高的准确率。Johannes Gunther 等人在焊接过程中使用 Actor-Critic 算法结合 Nexting 算法来实时控制焊接的功率，并使用激光焊接模拟器进行了多次初步控制演示[54]。

此外还有些研究提出了一些新的控制算法。芦川考虑到 DQN、Sarsa 等算法主要作用于离散型输出场景，并不能直接应用于文本所研究的连续环角焊缝的焊接作业，提出一种基于传统轨迹规划经验库和奖励逼近的轨迹规划方法 B-DPPO，首次将分布式近似策略优化 DPPO 运用于锅筒未知环角焊缝焊接中。并利用传统轨迹规划经验库，减少 B-DPPO 策略的无效探索，增强策略的学习效率。同时，为实现全位置焊缝不同位置焊缝成形优良采用近似策略优化将对流管与机器人空间位置和相对方向纳入一种新型方位奖励函数中，减少无谓探索计算[77]。

## 3 神经网络在焊接领域应用研究的发展趋势

随着人工智能技术的不断发展，焊接领域的智能化与自动化也逐渐加深。但由于大多数较为先进的技术均为近些年刚刚提出，在焊接领域中应用尚不广泛，因此焊接领域中所使用的控制与预测技术大多滞后于前沿的人工智能研究。因此，未来焊接领域将会结合较新的智能化技术，以达到更高的智能化水准，实现焊接的高度自动化。

### 3.1 神经网络在焊接过程预测中的发展趋势

这几年的人工智能研究大多聚焦于深度学习，而深度学习的核心就是神经网络。虽然近年来神经网络在焊接的预测与控制中已经得到了较为广泛的应用，但大多数的应用仅限于较为简单的任务，只需要浅层神经网络即可完成。而在复杂的预测过程大多使用的是一些传统的算法，少有利用深层神经网络的相关研究。近年来深度学习的发展使得神经网络的拟合能力大大提高，很多大型复杂任务可以通过神经网络端到端解决，因此在一些复杂任务上运用深层神经网络是焊接领域的发展趋势。

![Figure 3.1](../_media/paper/Figure3.1.svg)

图 3.1 焊接过程预测相关状态信息

图 3.1 展示了焊接过程预测中主要的状态信息，各个状态紧密连接，但各个连接中又充斥着复杂的变化过程，很难使用简单的数学公式来进行拟合。国内外现阶段在线预测的主要研究方向聚焦于通过熔池图像提取熔池形貌参数、通过焊接工艺参数来预测焊缝的尺寸以及缺陷检测任务。

神经网络已经广泛应用于通过焊接工艺参数来预测焊缝尺寸的过程中，由于该过程涉及到的参数量较少，一般浅层神经网络已经能够很好地完成该任务。另外，为了进一步提高神经网络的准确率，算法的优化过程大多使用遗传算法或者同时使用遗传算法与 BP 算法，国内的相关研究已经非常成熟。

现阶段常用的熔池图像处理手段有阈值分割、边缘检测算子、数学形态学等方式，这些方式往往繁杂且精度较低，识别效果一般。而在近年来的深度学习研究中，图像识别与检测技术大多使用的是卷积神经网络，它拥有极高的识别性能，在很多任务上已经超过了人类的水平。而且它可以高度并行化地处理数据，使得识别速度更快，明显提高了熔池图像提取的效果。现阶段国内外少有使用卷积神经网络直接对熔池图像进行提取的研究，因此使用卷积神经网络在熔池图像提取应用中将会逐渐取代传统算法。

缺陷检测也是焊接过程中研究的热门方向，传统方法大多也是使用简单的机器学习算法。但由于它多数情况下它的输入数据也是图片，数据处理难度也是非常之高。现阶段卷积神经网络在目标分类以及目标定位任务上也已经取得了突破性的成果，缺陷检测的任务非常适合使用卷积神经网络来处理。近年来，缺陷检测的任务也逐步使用卷积神经网络进行端到端训练来替换繁琐的图片处理和浅层神经网络结合的方式，因此深度神经网络在焊接中的应用是未来的大势所趋。

由于焊缝尺寸与接头力学性能之间的关系也是一个比较难以直接推导的过程，因此在该过程中使用神经网络进行拟合也是非常普遍的。该过程所涉及的参数也比较少，因此一般浅层神经网络已经能够较好地完成该任务。另外也有一些研究直接通过焊接参数或者环境图像来预测接头力学性能，这些都可以通过神经网络来完成。

### 3.2 神经网络在焊接过程控制中的发展趋势

![Figure 3.2](../_media/paper/Figure3.2.svg)

图 3.2 焊接过程控制主要流程

而在焊接的动力控制过程中，现阶段大多使用的是传统的 PID 控制器方法，通过闭环控制来达到机械臂的动力平衡。而近年来深度学习使强化学习的学习能力再次提高，能够应对高复杂度场景以及高复杂度的决策问题。另外，强化学习可以更加直接地给出在给定情形下所应采取的行为，且运算效率更高，因此该算法可以显著提高控制过程中的不稳定性。近年来强化学习在焊接过程中的研究非常少，有些研究是在 PID 的控制中加入强化学习辅助控制过程，提高了控制的稳定性[64]。

在焊接的工艺控制过程中，一些研究使用的是通过工艺参数来预测焊缝尺寸或者焊接接头的力学性能，这些可以采用上述焊接结构以及性能的预测方法，进而反向搜索在最佳焊接质量下所需要的工艺参数。该方法简单直观，但由于有着较为复杂的搜索过程，因此性能一般不高。而强化学习不仅可以应用在焊接动力控制过程中，只要设计合适的奖励函数，就可以应用在任何合适的任务上。在焊接的工艺控制中，目标是获得较好的焊缝质量，它可以作为强化学习中所使用的奖励值，而工艺参数的设置就可以当作是强化学习中所选取的动作选项，所以焊接工艺控制也非常适合使用强化学习来进行[54]。国内外该方法的相关研究更是少之又少，因此强化学习的广泛应用将会是焊接过程控制研究中的一大趋势。

## 结论

近年来随着深度学习的发展，神经网络作为一种热门的模型在各种行业中被广泛应用。神经网络相比于其它机器学习算法有着更强的非线性拟合能力，而且可以通过多层叠加以达到更高的性能，这使得深度学习中出现一些新的模型，比如卷积神经网络、循环神经网络等等。卷积神经网络已经广泛应用于计算机视觉领域，并且在分类与识别任务上已经超过了人类的水平。另外，强化学习也在与深度学习不断交融与发展，近年来很多任务使用从零开始的强化学习方法已经能够超过一些监督学习方法，而且强化学习在适用性上远高于监督学习，这使得强化学习相关技术也在近年来飞速发展。

焊接过程是一个高度复杂的过程，从焊接工艺参数到熔池的形成，再到焊接接头形成特定的尺寸，最后到焊接接头呈现不同的力学性能，包含了多种复杂的物理与化学变化，如果想要使用数学公式来对该过程进行描述，需要工程学、力学等学科的严密推导，而这个过程往往难以进行。为了能够更加方便地模拟这些过程，就需要使用一些机器学习算法来进行拟合。由于神经网络非常适合非线性函数的拟合，因此在焊接预测过程中得到了广泛的应用。

最近几年，焊接的过程控制一直在智能化方向不断发展，我们可以在焊接过程中对不同焊接工艺参数下的性能进行预测，进一步从中找到最佳的性能对应的焊接工艺参数，使用该参数获得更佳的焊缝质量。另一方面，焊接过程也在逐渐趋向完全自动化，这就需要焊接机器人能够自行捕捉焊缝区域并进行跟踪，而该过程需要通过计算机视觉技术来对目标识别和定位，这往往需要卷积神经网络的帮助。另外，焊接机械臂的控制也需要一定智能算法的辅助，传统方法大多使用的是 PID 控制器对焊接动力系统进行控制，但 PID 的控制往往无法适应焊接这一高度复杂的控制环境，因此可以通过神经网络对该控制过程进行辅助。另外，由于强化学习拥有着优秀的探索能力，并能够训练出较好的决策函数，因此强化学习也可以应用在焊接的动力以及工艺的控制过程中。但现阶段强化学习在焊接领域的应用尚少，因此强化学习在焊接控制过程的广泛应用是焊接领域发展的一大趋势。

## 致谢

时光荏苒，四年的大学生活转瞬而逝，值此论文完稿之际，不禁感慨万分。在这四年中，有着数不清的人给予了我帮助，我论文的完成离不开你们的支持，此时此刻，要感谢的人实在太多。

我要感谢我的指导教师祝美丽老师以及同组的张兆栋老师、王红阳老师与李锦竹学长。祝老师在我的论文写作过程中对我的论文进度进行监督与指导，为我的论文提出了宝贵的意见。张老师在焊接领域有着很深的造诣，曾教授过我焊接理论知识，并在论文的写作过程中通过视频会议的方式进行不断督促，为我的论文排版等问题进行了贴心的建议。王老师不仅对焊接领域有着深入的研究，还在机器学习领域颇有建树，曾教授过我机器学习在焊接机器人中的应用知识，并对我的论文结构与内容进行了悉心的讲解与指导。李锦竹学长在论文工作开始之初为我的论文课题进行悉心讲解，让我能够快速把握课题的研究方法，并让我了解到机器学习中常用的浅层神经网络与优化方法，对我的论文结构进行了悉心修正。

我要感谢为我深度学习以及机器学习启蒙的吴恩达老师与李宏毅老师。吴恩达老师的深度学习课程深入浅出、浅显易懂，为我的深度学习理论打下了坚实的基础，让我对人工智能有了全新的认识。李宏毅老师的课程诙谐幽默且与时俱进，通过有趣的例子为我讲解了深度学习与机器学习算法理论，并使我了解到最前沿的深度学习研究。

我要感谢材料科学与工程学院的各位曾教授过我的老师，各位老师的传道授业使我能够对材料科学有着一个较为全面的了解，让我能够顺利完成这四年的本科学业。我要感谢计算机科学学院各位曾教授过我的老师，是各位老师的耐心讲解使我对计算机底层技术有着深入的了解与兴趣，并让我能够对计算机程序设计有着更好的把握。我要感谢创新创业学院的刘胜蓝老师与刘洋老师，两位老师对我的深度学习以及计算机视觉的学习给予了肯定，让我更加坚定了自己深入深度学习研究的决心。

我要感谢我的室友陈威、许佳晨、王范旭，你们让我渡过了愉快的大学四年时光，并对我的学习不断提供鼓励。我要感谢我的挚友张经纬，在我的论文写作期间不断为我开导与鼓励，在精神上给予我了最大的支持。
我要感谢我的家人，在我的大学四年生活中为我默默支持与付出，为我提供了良好的学习环境。在我论文的写作过程中给予我关怀与照顾，让我能够专心进行论文的写作。
最后，我想用“因为热爱，所以期待”来为这四年的本科学习生活画上一个句号，期望今后的学习道路能够像之前一样充满动力，同时也企盼人工智能技术能够不断突破桎梏，为人类社会赋能。


# Refs

1. 王田苗,陶永.我国工业机器人技术现状与产业化发展战略[J].机械工程学报,2014,50(09):1-13.
2. 邱葭菲,邹金桥,王瑞权.TIG打底焊工艺研究[J].热加工工艺,2012(19):182-183.
3. 霍厚志,张号,杜启恒,等.我国焊接机器人应用现状与技术发展趋势[J].焊管,2017(2):36-42.
4. 孟宪伟,肖玉龙,唐宇佟,等.焊接智能化的研究现状及应用[J].电焊机,2019(9):84-87.
5. 周志华.机器学习[M].北京:清华大学出版社,2016.
6. McCulloch W S, Pitts W. A logical calculus of the ideas immanent in nervous activity[J]. The bulletin of mathematical biophysics, 1943,5(4):115-133.
7. Zhang SQ, Zhou ZH. Flexible Transmitter Network[J/OL]. Arxiv, 2020[2020-04-10]. https://arxiv.org/abs/2004.03839.
8. Rumelhart D E., Hinton G E, Williams R J. Learning representations by back-propagating errors[J]. nature, 1986,323(6088):533-536.
9. Hinton G E, Srivastava N, Krizhevsky A, et al. Improving neural networks by preventing co-adaptation of feature detectors[J/OL]. Arxiv, 2012[2017,01,28]. https://arxiv.org/abs/1207.0580.
10. Hinton G E, Salakhutdinov R R. Reducing the dimensionality of data with neural networks[J]. science, 2006, 313(5786):504-507.
11. Erhan D, Bengio Y, Courville A, et al. Why does unsupervised pre-training help deep learning?[J]. Journal of Machine Learning Research, 2010:625-660.
12. Glorot X, Bengio Y. Understanding the difficulty of training deep feedforward neural networks[C]. In Proceedings of the thirteenth international conference on artificial intelligence and statistics, Klagenfurt, Austria, 2010:249-256.
13. Vinod N, and Hinton G E. Rectified linear units improve restricted boltzmann machines[C]. Proceedings of the 27th international conference on machine learning (ICML-10), Haifa, Israel, 2010:807-814.
14. Maas A L, Hannun A Y, Ng A Y. Rectifier nonlinearities improve neural network acoustic models[C]. In Proc. ICML2013, Atlanta, United States, 2013, 30[1]:3.
15. He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]. Proceedings of the IEEE conference on computer vision and pattern recognition(CVPR-2016), Las Vegas, NV, USA, 2016:770-778.
16. Fukushima, K. Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position[J]. Biological Cybernetics, 1980,36:193-202.
17. LeCun Y, Bernhard E B, John S D, et al. Handwritten Digit Recognition with a Back-Propagation Network[C]. Advances in Neural Information Processing Systems,Denver,1990:396-404.
18. LeCun Y, Bottou L, Bengio Y, et al. Gradient-Based Learning Applied to Document Recognition[J]. Proceedings of the IEEE, 1998,86(11):2278-2324.
19. Zeiler M D, Fergus R. (2014, September). Visualizing and understanding convolutional networks[C]. In European conference on computer vision, Springer, Cham, 2014:818-833.
20. Krizhevsky A, Sutskever I, Hinton G E. ImageNet classification with deep convolutional neural networks[C]. Advances in Neural Information Processing Systems, Curran Associates RedHook, NY, USA, 2012:1097-1105.
21. Christian S, Liu W, Jia Y, et al. Going deeper with convolutions[C]. In Proceedings of the IEEE conference on computer vision and pattern recognition(CVPR-2015), Boston, MA, USA, 2015:1-9.
22. Karen S, Zisserman A. Very deep convolutional networks for large-scale image recognition[C]. The International Conference on Learning Representations (ICLR), San Diego, United States, 2015:1-14.
23. Ioffe S, Szegedy C. Batch normalization: Accelerating deep network training by reducing internal covariate shift[J/OL]. Arxiv, 2015[2015-03-02]. https://arxiv.org/abs/1502.03167.
24. Szegedy C, Vanhoucke V, Ioffe S, et al. Rethinking the inception architecture for computer vision[C]. In Proceedings of the IEEE conference on computer vision and pattern recognition(CVPR-2016), 2016:2818-2826.
25. Chollet F. Xception: Deep learning with depthwise separable convolutions[C]. In Proceedings of the IEEE conference on computer vision and pattern recognition(CVPR-2017), 2017:1251-1258.
26. Szegedy C, Ioffe S, Vanhoucke V, et al. Inception-v4, inception-resnet and the impact of residual connections on learning[C]. In Thirty-first AAAI conference on artificial intelligence, 2017.
27. He K, Zhang X, Ren S, et al. Identity mappings in deep residual networks[C]. In European conference on computer vision, Springer, Cham, 2016:630-645.
28. Xie S, Girshick R, Dollár P, et al. Aggregated residual transformations for deep neural networks[C]. In Proceedings of the IEEE conference on computer vision and pattern recognition, Honolulu, HI, USA, 2017:1492-1500.
29. Huang G., Liu Z., Van Der Maaten L., et al. Densely connected convolutional networks[C]. In Proceedings of the IEEE conference on computer vision and pattern recognition(CVPR-2018), Salt Lake City, UT, USA, 2018:4700-4708.
30. Hu J, Shen L, Sun G. Squeeze-and-excitation networks[C]. In Proceedings of the IEEE conference on computer vision and pattern recognition(CVPR-2018), 2018:7132-7141.
31. Howard A G, Zhu M, Chen B, et al. Mobilenets: Efficient convolutional neural networks for mobile vision applications[J/OL]. Arxiv, 2017[2017-01-17]. https://arxiv.org/abs/1704.04861.
32. Zoph B, Vasudevan V, Shlens J, et al. Learning transferable architectures for scalable image recognition[C]. In Proceedings of the IEEE conference on computer vision and pattern recognition, Salt Lake City, UT, USA, 2018:8697-8710.
33. Tan M, Le Q V. Efficientnet: Rethinking model scaling for convolutional neural networks[J/OL]. Arxiv, 2019[2019-11-23]. https://arxiv.org/abs/1905.11946.
34. Radosavovic I, Kosaraju R P, Girshick R, et al. Designing Network Design Spaces[J/OL]. Arxiv 2020[2020-03-30]. https://arxiv.org/abs/2003.13678.
35. Degris T, White M, Sutton R S. Off-policy actor-critic[J/OL]. Arxiv 2012[2013-01-20]. https://arxiv.org/abs/1205.4839.
36. Mnih V, Badia A P, Mirza M, et al. Asynchronous methods for deep reinforcement learning[C]. In International conference on machine learning, New York City, United States, 2016:1928-1937.
37. Mnih V, Kavukcuoglu K, Silver D, et al. Playing atari with deep reinforcement learning[J/OL]. Arxiv 2013[2013-12-19]. https://arxiv.org/abs/1312.5602.
38. Silver D, Huang A, Maddison C J, et al. Mastering the game of Go with deep neural networks and tree search[J]. nature, 2016,529(7587):484.
39. Silver D, Schrittwieser J, Simonyan K, et al. Mastering the game of go without human knowledge[J]. Nature, 2017,550(7676):354-359.
40. 裴浩东,樊丁,马跃洲,等.人工神经网络及其在焊接中的应用[J].甘肃工业大学学报,1996(1):1-6.
41. 李航.统计学习方法[M].北京:清华大学出版社,2019.
42. 顾盛勇. 基于视觉传感的高效深熔锁孔TIG焊焊缝识别及熔透状态的研究[D].广州:华南理工大学,2018.
43. 贺峰,史亚斌,王锋,等.基于多传感器和支持向量机的GMAW焊接过程模式识别研究[J].科技创新与应用,2018(34):1-4+7.
44. Zhifen Zhang. Real-time seam defect identification for Al alloys in robotic arc welding using optical spectroscopy and integrating learning mechanisms in 20kw-class CO2 laser welding processes[J]. Journal of Physics D:Applied Physics, 2002, 36(2): 192.
45. Zhu H, Ge W, Liu Z. Deep Learning-Based Classification of Weld Surface Defects[J]. Applied Sciences, 2019,9(16):1-10.
46. Shojaeefard M H,Behnagh R A,Akbari M,et al. Modeling and pareto optimization of mechanical properties of friction stir welded AA7075 / AA5083 butt joints using neural network and particle swarm algorithm[J]. Materials & Design, 2013, 44(2):190198.
47. 赵成涛.基于BP神经网络的镁合金焊缝成形预测[J].热加工工艺,2019(15):178-182.
48. 黄军芬,薛龙,黄继强,等.基于视觉传感的GMAW熔透状态预测[J].机械工程学报,2019(17):41-47.
49. Yin L, Wang J, Hu H, et al. Prediction of weld formation in 5083 aluminum alloy by twin-wire CMT welding based on deep learning[J]. Welding in the World, 2019, 63(4):947-955.
50. Liu B, Jin W, Lu A, Liu K, et al. Optimal design for dual laser beam butt welding process parameter using artificial neural networks and genetic algorithm for SUS316L austenitic stainless steel[J]. Optics and Laser Technology,2020,125.
51. 张喆,张永林,陈书锦.基于遗传BP神经网络的搅拌摩擦焊温度模型[J].热加工工艺,2020,49(03):142-145.
52. Ding H, Wang Z, Guo Y, et al. Research on laser processing technology of instrument panel implicit weakening line based on neural network and genetic algorithm[J]. Optik,2020,203.
53. Zhang Y, You D, Gao X, et al. Real-time monitoring of high-power disk laser welding statuses based on deep learning framework[J]. J Intell Manuf 31, 2020:799-814.
54. Johannes G, Patrick M P, Gerhard H, et al. Intelligent laser welding through representation, prediction, and control learning: An architecture with deep neural networks and reinforcement learning[J]. Mechatronics,2016,34.
55. Krizhevsky A, Sutskever I, Hinton GE. ImageNet classification with deep convolutional neural networks[C]. Advances in Neural Information Processing Systems, Curran Associates RedHook,NY,USA,2012:1097-1105．
56. 刘新锋. 基于正面熔池图像和深度学习算法的PAW穿孔/熔透状态预测[D].济南:山东大学,2017.
57. 覃科,刘晓刚,丁立新.基于卷积神经网络的CO2焊接熔池图像状态识别方法[J].焊接,2017(06):21-26+70.
58. 焦敬品,李思源,常予,等. 集箱管接头内焊缝表面缺陷识别方法研究[J]. 仪器仪表学报,2017,38(12):3044-3052.
59. 杨志超,周强,胡侃,等.基于卷积神经网络的焊接缺陷识别技术及应用[J].武汉理工大学学报(信息与管理工程版),2019,41(01):17-21.
60. 刘梦溪,巨永锋,高炜欣,等.焊缝缺陷图像分类识别的深度置信网络研究[J].测控技术,2018,37(08):5-9.
61. 刘梦溪,巨永锋,高炜欣,等.深度卷积神经网络的 X 射线焊缝缺陷研究[J].传感器与微系统,2018,37(05):37-39.
62. Hou W, Wei Y, Jin Y, et al. Deep features based on a DCNN model for classifying imbalanced weld flaw types[J]. Measurement, 2019, 131:482-489.
63. 黄旭丰. 基于深度迁移学习的焊接质量在线监测方法研究[D].南宁:广西大学,2019.
64. 张永志,董俊慧,侯继军.广义动态模糊神经网络焊接接头力学性能预测[J].焊接学报,2017,38(08):37-40+130.
65. 刘政军,张琨,刘长军.基于双重人工神经网络模型预测焊接接头强度系数的研究[J].兵器材料科学与工程,2018,41(05):53-56.
66. 刘立鹏,王伟,董培欣,等.基于遗传神经网络的焊接接头力学性能预测系统[J].焊接学报,2011(7):105-108.
67. 张昭,白小溪,李健宇.基于大数据驱动的焊接接头力学性能预测[J].电焊机,2020,50(04):75-78+139.
68. Dengkui Zhang, Guoqing Wang, Aiping Wu, et al. Study on the inconsistency in mechanical properties of 2219 aluminium alloy TIG-welded joints[J]. Journal of Alloys and Compounds,2019,777: 1044-1053.
69. 阮德重,张登魁,王国庆,等.基于RBF神经网络预测2219铝合金多层TIG焊接头拉伸性能研究[J].焊接技术,2019,48(06):22-27.
70. 张抱日,顾盛勇,石永华.基于焊缝熔透检测的机器人深熔K-TIG焊接系统[J].机械工程学报,2019,55(17):14-21.
71. 陈皓,堵俊,葛佳盛等.一类多目标焊缝跟踪优化模型和算法[J]. 制造业自动化, 2015, 37(8):1-3.
72. 刘建昌,苗宇. 基于神经网络补偿的机械臂轨迹控制策略的研究[J].控制与决策,2005,20(7):732-736.
73. 张文辉,齐乃明,尹洪亮. 自适应神经变结构的机器人轨迹跟踪控制[J].控制与决策, 2011,26(4):597-600.
74. 王保民,张明亮.基于RBF神经网络的弧焊机器人轨迹跟踪控制方法[J].兰州理工大学学报,2019,45(03):85-89.
75. 李广创,程良伦.基于深度强化学习的机械臂避障路径规划研究[J].软件工程,2019,22(03):12-15.
76. 刘卫朋. 焊接机器人焊接路径识别与自主控制方法研究及应用[D].天津:河北工业大学,2016.
77. 芦川. 锅炉内壁管板焊接跟踪智能焊接机器人机构设计与运动控制[D].湘潭:湘潭大学,2019.
