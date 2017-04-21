---
title: Highway network
layout: post
---
reference:http://www.tuicool.com/articles/F77Bzu
这几天，关于 ICCV 2015 有一个振奋人心的消息——“微软亚洲研究院视觉计算组的研究员们凭借深层神经网络技术的最新突破，以绝对优势获得 图像分类、图像定位以及图像检测 全部三个主要项目的冠军。同一时刻，他们在另一项图像识别挑战赛 MS COCO （ Microsoft Common Objects in Context ，常见物体图像识别）中同样成功登顶，在图像检测和图像分割项目上击败了来自学界、企业和研究机构的众多参赛者”。

这个巨大的突破到底是怎样实现的？作者 Kaiming He 还做了哪些相关工作？ICCV 2015 上还有哪些有趣的论文？ 今天我就来介绍一二。 它们分别是：

《 Deep Residual Learning for Image Recognition 》 . 2015. arXiv pre-print.

《 Training Very Deep Networks 》. Neural Information Processing Systems (NIPS 2015 Spotlight).

《 Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification 》 . ICCV 2015.

《 Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books》. ICCV 2015.

《Deep Fried Convnets》 . ICCV 2015.

《Fast R-CNN》 . ICCV 2015.

Deep Residual Learning for Image Recognition

首先第一篇肯定要谈，开创了152层 deep network，成功摘取 ILSVRC2015 全（主要）类别第一名 的工作，来自 MSRA 的 Deep Residual Network 。虽然 residual network 的名字很新，但 其实可以把它看成是我已经介绍和推荐过无数次的 Highway Networks 的特例 （Highway Networks 的介绍见下）。尽管如此，作者在解释 residual 的 motivation 时还是非常充分合理，让人不得不佩服其在背后自己的思考。

                          
作者提出 residual network 的 motivation 其实依然是 information flow 在 deep network 中受阻。大家都可以想象到这个问题。但是这篇工作中，作者是如何“验证”这个问题的呢？他们用了两种“角度”， 第一种 是在 Introduction 里就提到的：当他们用 deep network 和可类比的 shallower network 横向对比（如上图），发现 deep network 的 training error 总是比 shallower network 要高。可是这在理论上是不对的。因为如果一个 shallow network 可以被训练优化求解到某一个很好的解（subspace of that of deep network），那么它对应的 deep 版本的 network 至少也可以，而不是更差。但事实并不是如此。这种 deep network 反而优化变差的事情，被作者称为“degration problem” 。 


第二种 角度是在实验部分提到的，从理论角度分析这种 degration 不应该是 gradient vanishing 那种问题，而是一种真正的优化困难。于是，为了解决这个 degration problem，作者提出了 residual network，意思是说，如果我们直接去逼近一个 desired underlying mapping (function) 不好逼近（优化困难，information flow 搞不定），我们去让一个 x 和 mapping 之间的 residual 逼近 0 应该相对容易。

这就是作者给 highway networks 找的 residual 的解释。



那么，在实际上，residual network block（上图）就相当于是 Highway network block 中 transform gate 和 transform function 被特例后的结果——因为是特例了，也自然 reduce 了 parameter，也算是这篇工作中作者一个卖点。
现在问题是为什么是Highway Networks 的特例呢？
介绍完 Highway Networks 就明白了。
Training Very Deep Networks

好了，现在让我们来再介绍一遍 Highway Networks 这个工作（我第一次介绍这个工作还是在今年 9月18日，比现在早三个月，骄傲脸）。为了更好的对比和更大力的推荐，以下笔记是我重新看了论文和代码后，重写的（不是以前的 copy-paste 版本了）。这篇论文前身是《Highway Networks》，发表于 ICML workshop 上。最初放在 arXiv 上，现在已经被 NIPS'15 接收。这个工作纯被 LSTM 的 gate 启发，既然 gate 也是为了解决 Information flow，有没有其他方式去解决？更直观一点的，不通过 gradient 的？既然 information 像被阻隔了一样，我们就“暴力”让它通过一下，给它们来个特权——在某些 gate 上，你就不用接受“审查”（transform）了，直接通过吧。这像高速公路一样——于是就有了这个名字，Highway Networks（HW-Nets）。

To overcome this, we take inspiration from Long Short Term Memory (LSTM) recurrent networks. We propose to modify the architecture of very deep feedforward networks such that information flow across layers becomes much easier. This is accomplished through an LSTM-inspired adaptive gating mechanism that allows for paths along which information can flow across many layers without attenuation. We call such paths information highways. They yield highway networks, as opposed to traditional ‘plain’ networks.

加粗了 adaptive，这就是这个 mechanism 的重点 ， 也是之所以说 Deep Residual Network 是它的一个特例的原因所在 。 在文章中，公式（2-3）就是他们的机制。

公式（3）是公式（2）变形。 核心是 transform function H 和 transform gate T 。这样，对于当前的这个 input，在这个 layer 里，公式（3）决定多大程度是让它去进行 nonlinear transform（隐层），还是多大程度上让它保留原貌 直接传递给下一层，直通无阻。

 
那么，在这里，其实我们也可以把这个公式（3）像 residual block 一样，拆成两个 component，一个是 H，一个是 x。如果是 x，就变成了 residual block 中 identity mapping 的 skip connection。
这是一个 intuition 的解释。那么再具体等价一下，Deep Residual Network 那篇讲到，自己比 Highway Networks 的一个优势是，parameter 少，并且自己的“gate”永远不 close。
    这两个优势，既都对，也都不对。  
    关于第一点 ，这是事实，而这恰恰是把 Highway Networks 的一个优势给抹掉了。在 Highway Networks 文章的 4.1 部分，有讨论自己这种 adaptive mechansim for information flow learning 的优点。也就是说，如果说 Highway Networks 是 data-independent 的，那么 Deep Residual Network 就是 data-dependent 的，不 flexible，更可能不“最优”（Deep Residual Network 就是 Highway Networks 中提到的 hard-wired shortcut connections）， 它只是一种 p=0.5 的二项分布的期望（np） 。关于第二点 ，就不一定是事实了。因为实际上，虽然公式（3）看起来，transform gate 有可能完全 close，然而 transform function 却永远不可能为 0 或者 1，它的取值范围只是 (0,1) 开区间，所以在这点上它和 Deep Residual Network 是一样的。
对比说完，继续介绍 Highway Networks。Highway Networks 更巧妙的是，它在公式（3）这种 layerwise 的基础上，进一步拆分， 把每一个 layer 的处理从 layerwise 变成了 blockwise ，也就是我们可以对于 input 中的某一部分（某些维度）选择是 transform 还是 carry。这个 blockwise 的实现也在实验中被验证特别有用，大家可以去看下，我就不重复粘贴了。

Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification

再来介绍一篇 Kaiming He 的工作，虽然工作最后的结果没有第一篇那么极其厉害璀璨耀眼，但依然也是效果斐然。这篇工作的入手点非常“稳准狠”，就是大家都这么爱用 ReLU nonlinear transform，这 东西的缺点是啥？能不能改进？然后它们就分析，这东西主要有俩缺点，一个是它 压根不是对称 的，一个是这东西 transform 后的东西还是不对称的，无论你怎么假设 input/weight distribution 是对称的，还是没用。这俩缺点一起，就会影响 neural networks 的 convergence。于是，既然这东西这么不好用，不如我们把它改造下，它不对称，我们就让它稍微对称点？就有了 Parametric ReLU （PReLU） 的工作。来对比一下 ReLU 和 PReLU：

    配合公式（1）可以看出，PReLU 其实是调整了 x<0 时 的斜率。为什么这样调整呢，其实这个调整可以被看成一种“中和”（offset），既然你不对称，你的 mean 就不是 0。我就可以尽量让你的 positive mean 趋近于 0。这个 intuition 也被他们用数据证明了一下：
那么有了这种改造后，他们解决了啥问题呢。第一是，确实这样改造后，收敛变好了，那么效果也就变好咯——就有了题目这个结果。第二是，他们想探究一下 initilization for ReLU or PReLU 的理论，然后探究了挺多，虽然没有实质的帮助，但是得出了一些中间产物吧。 那么这种改造的优点还有啥呢，这个首先也是 blockwise 的（跟 Highway Networks 一样），其次也是 differentiable 的，可以融入各种 end-to-end 里。

Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books

这篇工作是典型的 multi-modall，但是不是一张 image 和一句 caption 的 align 了，而是变成了多幅 image 和一个 dialogue 的 align——book-movie align。



上面就是这个工作 project 页面给出的 example，来自去年大火的《消失的爱人》。这部电影中，女主角会自己写日记，在电影中也用日记作为了旁白。所以很符合电影和书的 align 设定。在这篇论文的工作中，他们用电影的多幅截图（shots）当做 movie units，并用相对应的字幕（subtitles）作为 dialogue。值得一提的是，这篇工作因为用到的是 sentence level 的 similarity，所以需要 sentence embedding，而 sentence embedding 就是用到的他们自己组内的 Skip-Thought Vectors 的工作。

那么这样的 alignment 能用来做什么呢？可以反过来利用呢，就是给出 shot，我用书里的一些话，作为 shot（image）caption。比如这样：



Deep Fried Convnets

这篇工作的名字很新潮，很有食欲 =___,= 因为它源自于一个叫 Fastfood kernel 的工作，一 deep，就变成 fried 了。以上纯属胡扯。这个工作的 motivation 其实非常 general，和现在的 compressing neural networks 什么的各种工作都紧密相关。大概就是说，deep neural networks 里的参数太多了，所以 computation 和 storage 都很高；我们有没有可能能通过只 store 一小部分 parameters 来 predicting/recover 其他的 parameters 呢？

这个工作的 fundamental 的思想也是这个。具体上，他们是将 convnets 里面的 fully connected layer 用 Adaptive Fastfood transform 给 replace 掉。在 convnets 里面，这个 fully connected layer 是最耗费时间空间计算存储的，而这个 transform 则可以通过几个矩阵和几个高效的运算（而且是完全 back propagation 的），在不降低 convnets 的学习能力的情况下，实现提速和节省空间。与此同时，这个 Adapative Fastfood transform 是一个 composite component，所以也可以用 dropout 这些技巧。 这篇工作的思想大概就是这样，数学上我就不在这里贴公式了，不过推荐一下这篇算法部分介绍的写作。这个写作顺序，也是给一个大的 general 框架，再给出各种相关的 intuition reference。其实仔细思考会发现，这个框架并不 novel，但是这样写，就避免了读者被各种 distracted 和读不懂。

最后再说下这个工作比较大的 limit ，这个也在他们实验里表现出来了。就是这个 transform 如果想表现好（其实就是“predicting”好），需要依赖“喂”给它大量 features。这个 limit 也是这些作者的下一个工作的出发点，有兴趣可以去看一下《ACDC: A Structured Efficient Linear Layer》。

Fast R-CNN
这篇论文的作者是大神 Ross B. Girshick（业界代号“RBG”），其在博士期间就获得了 Pascal VOC 终生成就奖；这篇一看就是改进 R-CNN 的工作，然而其实 R-CNN 也是他自己做的。文章署名只有他一人。

这一系列 object detection 的工作，主要 解决和面临的困难是 ：object detection 比 image classification 难在：（1）有非常多 candidate object locations（也叫 proposal），太难一一检索；（2）即使是检索到一些 location，也会比较模糊，对第二步 detection 的工作很有影响。从上面这个叙述可以可能出，以前的 object detection 工作是 pipeline 的，先找 proposal，再 detect。pipeline 的工作一个是慢，一个是 error propagation 问题严重，最终任务效果很受底层任 subtask 的表现影响。Fast R-CNN 就是在 pipeline 这件事上，有了突破，就是我又做 classification，又做 detection ——不再 pipeline，而是同时输出两个结果，也就是 joint training 了。

那么是如何实现的呢，先 selective search 得到大概 2K 个 proposal（也叫 RoI，region of interests），然后用 invariant scale 得到图片金字塔，最后在全连接得到特征时，用 hierarchical sampling 的方式，共享特征（reduce computation）给两个新的全连接，最后是两个 sibling 的优化目标。 第一个优化目标是分类，使用 softmax ，第二个优化目标是 bbox regression。对比以前 MSRA Kaiming He 的 SPP-Net 工作，其实可以认为是一个 joint training 版本的 SPP-Net，把 classifier 也一起搞进来（不用再单独训练一个 SVM）。

最后，推荐两个相关资源。一个是关于这个工作有一个很精彩的 CVPR 2015 slides，叫《 Fast R-CNN—— Object detection with Caffe 》，涵盖了 zooming in networks 内部，还分析了源码。另一个是，这篇文章在知乎上有非常详实的讨论，大家都追本溯源，加入了许多个人理解，很精彩，对这个工作有兴趣的同学请移步知乎《 如何评价 rcnn 、 fast-rcnn 和 faster-rcnn 这一系列方法？ 》。

其实小S 看 ICCV 的感受是，这个会的工作还是非常 theory + practice 并存的。比如最后一个工作，Fast R-CNN 中，虽然理论上看起来 hierarchical sampling 会有问题，然而最后 practical 中并不存在。现在 theorical DL 的发展，肯定不可能是一帆风顺的，所以这种 theory/intuition driven 然后 practice 印证的道路是必不可少的。这种经验，既有些遗憾又很宝贵。大家一起加油吧。

其他相关内容，可回复橘色代码（如【 GH017 】）或者点击文章标题（已加内链）跳转阅读：

GH017 辨析计算机视觉、计算机图形学和数字图像处理

GH025 Information Flow Mechanisms in LSTMs and their Comparison

GH030 Multi-modal Deep Learning 初窥

GH031 Improve Information Flow in Seq2Seq Learning