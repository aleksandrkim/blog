# 3D Semantic Segmentation with Submanifold Sparse Convolutional Networks
###### Benjamin Graham, Martin Engelcke, and Laurens van der Maaten (Facebook AI Research), _CVPR 2018_, [link to the paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Graham_3D_Semantic_Segmentation_CVPR_2018_paper.pdf)

The following content should be viewed as an easy to understand introduction to the topic of the paper and its high-level recap. It will, hopefully, also help you understand the whole paper if you choose to find out more details about this work.

---

What is semantic segmentation of an image? For **each** data point (in this case, a pixel) a class label needs to assigned, this means that our network needs to process each input point to output precise values. To avoid processing every pixel, attempts have been made to approximate per-point labels, but nowadays, the best results are achieved by pushing a full resolution image through a dense convolution-based neural network, e.g. [FCN](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) or [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

<p align="center">
  <img width="460" height="300" src="https://devblogs.nvidia.com/wp-content/uploads/2016/11/figure1.gif">
  <p align="center"> from https://devblogs.nvidia.com/image-segmentation-using-digits-5/ </p>
</p>

Such 2D semantic segmentation provides accurate labels for every pixel and is useful for robot navigation, decision making, and anything else that relies on fully understanding the environment. Deep learning has helped achieve great results in this area, but the same cannot be said about 3D semantic segmentation, where the input is not a 2D image, but a 3D representation of the real world, e.g. a voxel grid or a point cloud.

A pixel, which represents a square in an image, is the 2D counterpart of a voxel, which represents a cube in 3D space. So a **voxel grid** is simply a volumetric extension of an image to three dimensions. Thus, it presents an opportunity to extend 2D convolutional neural networks (CNNs) to 3D in hopes of replicating their success in new applications. Unfortunately, most models are too slow/difficult/heavy to train and deploy for actual applications.

As you know, a common 2D convolution is a square kernel that slides across every pixel in the image to produce dense output. This means that the number of operations required for a single convolutional layer (`stride=1, padding=1`) is proportional to the number of pixel in its input, for example, for a `600x800` image and a `3x3` filter you need `3x3x600x800` = 4.32 million operations. For semantic segmentation, each pixel needs to be treated individually to produce precise labels, so the complexity is `3x3xHxW` where H and W are the dimensions of the image. 

A simple extension to 3D is to make the convolutional kernel a cube and slide it across every voxel in the voxel grid, adapt existing CNN architectures with this new operator and output dense labels. This would make the complexity `3x3x3xHxWxDepth`, which is already a lot for a modestly sized grid. Keeping in mind that a typical CNN includes hundreds of convolutional layers, this complexity quickly becomes unsuitable for practical applications. 

<p align="center">
  <img width="460" height="300" src="https://miro.medium.com/max/1440/1*z5wfLwBCEgAPTI_U1mSaVA.png">
  <p align="center"> from https://towardsdatascience.com/understanding-1d-and-3d-convolution-neural-network-keras-9d8f76e29610 </p>
</p>

However, voxel grids come with a an interesting property that images do not have. A voxel grid containing a bunny will have a lot of empty voxels where there is nothing but air, and the active voxels (sites) containing the actual bunny only comprise a part of the whole grid. 3D neural networks can take advantage of this sparsity, and it is exactly what this paper aims to do with its new convolutional operator. The authors' objective is to improve the efficiency of the 3D convolutional operator employed for volumetric semantic segmentation without negatively impacting its performance to make bigger and more powerful architectures feasible to train and deploy.

To better understand the authors' contributions, we will quickly review a few related works, go over the proposed idea and its implementation, and look at the reported results. Finally, we will discuss the merits of the new operator and further examine its drawbacks to get a better understanding of where it should (and should not) be applied. 



### Related works
To understand what other approaches were explored in this direction and how this paper compares to others, we will look at a few previously published ideas. Below, to avoid any complications, some things related to operations on 3D voxels will be supported by 2D, not 3D, illustrations, but you can imagine how the same logic would work on higher dimensional input.


#### OctNet by G Riegler, A. O. Ulusoy, and A. Geiger [[1]](#1)
OctNets try to avoid processing empty space that does not contain any relevant information by representing a voxel grid via [octrees](https://en.wikipedia.org/wiki/Octree). Octrees are the three-dimensional extension of quadtrees and it essentially means that the voxels/cubes in a voxel grid are no longer of the same size. If a part of a grid only contains a single point but fits eight voxels, then an octree will represent that whole part with a single large cube instead of eight small voxels to conserve memory. Here is an example of how an OctNet would represent a sparse grid:

<p align="center">
  <img src="https://raw.githubusercontent.com/aleksandrkim/blog/master/attachments/octnet_both.png">
</p>

Normally, this grid would be comprised of `6x6` = 36 voxels, but an OctNet would see it as 7 big voxels and 8 regular voxels. This allows the network to perform fewer operations than usual, but still a significant amount - **for 2** active cells (sites) 7+8=**15 operations** are required.


#### Vote3Deep by M. Engelcke, D. Rao, D. Z. Wang, C. H. Tong, and I. Posner [[2]](#2)
The main idea behind Vote3Deep is to replace convolutions with a voting operator kernel and use a ReLU activation to produce sparse output. Similarly to upsampling, a voting operator takes a single input and produces multiple outputs, for example, to replicate a `3x3x3` convolution, a `3x3x3` voting tensor would be used to produce outputs for every single input. After applying a voting kernel to each active site the output is much denser than its input. To balance this, a non-positive bias is learned and applied and then a ReLU discards all negative output resulting in a sparser result.

<p align="center">
  <img src="https://raw.githubusercontent.com/aleksandrkim/blog/master/attachments/vote3deep.png">
</p>

With appropriate biases the output can stay sparse throughout a big network based on these voting operators, but, still, to produce it a lot of operations are performed - **for 2** active sites in the example, **16 operations** are needed.



#### Sparse Convolutions by B. Graham [[3]](#3)
Sparse convolutions are based on a simple idea - perform (activate) convolution only if at least one input cell in the receptive field is active, otherwise immediately output a zero state vector - the convolution is not active. 

<p align="center">
  <img src="https://raw.githubusercontent.com/aleksandrkim/blog/master/attachments/sc_active.png">
</p>

The complexity of this method is the same as **Vote3Deep** - **16 multiplications** **for 2** active sites; however, the sparsity is not maintained. Since every active cell activates multiple convolution filters, the output density increases with each layer. Below you can see how consecutive layers get more and more dense as the information spreads with each SC. This paper was also written by Ben Graham and the "Submanifold Sparse Convolutions" work is based on it.
 
<p align="center">
  <img src="https://raw.githubusercontent.com/aleksandrkim/blog/master/attachments/sc_sparsity.png">
</p>



### Main idea
A **submanifold sparse convolution** (SSC) is a sparse convolution (SC) with a stricter requirement for activation. An SC is active if anywhere in its receptive field there is an active site, but an SSC is active if and only if at the center of its receptive field there is an active site; in other words, if the central site is active. This small tweak is the only thing different between these two proposed operators, but it solves a major problem with SC - sparsity is now maintained when applying consecutive SSCs! 

<p align="center">
  <img src="https://raw.githubusercontent.com/aleksandrkim/blog/master/attachments/ssc_sparsity.png">
</p>

 Let's denote `a` the number of active sites in the receptive field, `m` and `n` the numbers of input and output channels respectively, and `d` the number of dimensions. There are only three possible input types: 
1. the central site is active, `a > 0`
2. the central site is not active, but at least one other site in the receptive field is active, `a > 0`
3. no site in the receptive field is active, `a = 0`

<p align="center">
  <img src="https://raw.githubusercontent.com/aleksandrkim/blog/master/attachments/input_scenarios.png">
</p>

Here is the number of floating-point operations (FLOPs) needed for a convolutional 3x3 kernel of a standard convolution (C), SC, and SSC:

| Input | C | SC | SSC |
| ---: | :---: | :---: | :---: |
| active, a > 0 | 3^d^mn | amn | **amn** |
| not active, a > 0 | 3^d^mn | amn | **0** |
| not active, a = 0 | 3^d^mn | 0 | **0** |

Clearly, SSC is the most computationally efficient operator and if it can provide the same accuracy as standard dense convolutions, this will be a big boost towards practical applications of 3D CNNs. To make powerful CNNs based on SSCs, one would also need activation functions, batch normalization, pooling, and transposed convolution.

Activation functions and batch normalization are defined as standard operations but are restricted to the set of active sites to avoid unnecessary computation.

Pooling (max and average) are defined as instances of SC, not SSC - pooling filters compute averages and max values considering only active cells within their receptive fields, ignoring empty spaces. Average pooling always divides by the number of all cells (e.g. 9 for `3x3` pooling), not only active ones. If pooling was defined similar to SSC, some information and invariance to translation would be lost since downsampling has to be dependent on the input's spatial size and not on the number of active sites it contains. SC-based pooling ensures each active site is considered. 

Transposed convolution is defined as an inverse SC.



### Implementation
To take full advantage of SSC, the algorithm to perform convolutions needs to be different from the standard one to avoid unnecessary operations.

Instead of sliding a convolutional kernel across the input's spatial dimensions, it is applied only to the set of active sites. In the case of images, where each pixel has information, the two approaches would be equivalent as each point in the input space is an active site. So mathematically nothing has changed, but now we can take advantage of data sparsity.

In addition to the whole input being sparse, the receptive field of each filter is also sparse, so instead of multiplying each weight of the kernel with its corresponding input cell (active or not) we can multiply individual weights only if their input is active.

These two modifications do not influence the output of a convolutional layer, only add conditions to its computations. 

Regular layers store features in tensors of the same shape as the input, but we can utilize a different data structure to save memory since not all cells contain features. A matrix of size `a x m` can be used to store feature vectors of length `m` of each active site (`a` in total), and a hash table of shape `a x 2` can be utilized to remember where each active site is located in the original input. 

All active sites are guaranteed to be multiplied with the central weight of a filter by definition of a SSC, but some of them can also be multiplied by non-central weights if a neighboring active site triggers a convolution that has multiple active cells in its receptive field (which is why the output location also needs to be remembered for every multiplication). To perform multiplications needed for convolution, a tensor can be employed to remember which filter weight is multiplied with which active input and where the result should be put afterward. For example, for a `3x3` kernel a `3x3x2` tensor is needed to store this information - a pair `(input_site, output_site)` for each cell in the filter.

<p align="center">
  <img src="https://raw.githubusercontent.com/aleksandrkim/blog/master/attachments/implementation.png">
</p>

Here is some pseudocode to get an intuition of how an SSC may be implemented:

```py
input_features - matrix, shape=(a, m)
output_features - matrix, shape=(a, n)
locations - hash table, shape=(a, 2)  # identical for input and output
W - matrix, shape(9, m, n)  # weight of a 3x3 kernel (example)
R - list of lists, shape(9, )  # each weight has a list to remember which sites need to be processed

# find which multiplications are needed
for input_site in range(a):  # go over all active sites
  all_output_sites = sites_with_receptive_field_covering(input_site)
  active_output_sites = filter_active(all_output_sites)  # active sites that have this input_site in their receptive field
  for output_site in active_output_sites:
    weight_i = cell_in_filter(output_site, input_site)  # which cell in the kernel covers this input_site
    R[weight_i].add((input_site, output_site))  # remember to what to multiply with this filter weight and where to put the result

# perform multiplications
for index, R_cell in enumerate(R):
  weight = W[index]
  for input_site, output_site in R_cell:
    output_features[output_site] += input_features[input_site].dot(weight)
```
The implementation for a SC would be similar, except the `filter_active` function would be less constrained. 



### Evaluation
To test large 3D CNNs based on SSCs, the authors replicated two popular semantic segmentation networks - FCN and U-Net with full convolutional layers being replaced by SC and SSC layers.

<p align="center">
  <img src="https://raw.githubusercontent.com/aleksandrkim/blog/master/attachments/fcn_unet_paper.png">
</p>

The results of these SSC-based networks are impressive across multiple benchmarks, most notably ShapeNet [[4]](#4) and ScanNet [[5]](#5) (on ScanNet's leaderboard an SSC-based CNN still holds the 2nd place). Not only are these networks efficient, but also more accurate than previous approaches. The graphs below show SSC-based CNNs achieving higher accuracy compared to other models under the same computational budget. More experimental results are available in the paper.

<p align="center">
  <img src="https://raw.githubusercontent.com/aleksandrkim/blog/master/attachments/shapenet.png">
</p>

### Drawbacks
While SSCs are a good alternative to standard dense convolutions and allow training of deep and powerful models thanks to efficient computations, they are not a cost-free solution. After spending all this time understanding why SSCs are great, we will take some time to explore their downsides:

1. The number of parameters of an SSC does not change compared to a dense convolution.
An SSC kernel has a weight for each of its cells regardless of how many times each cell has actually been activated. So even if fewer FLOPs are needed for the forward pass, the backward pass still has to optimize the same amount of weights. Of course, less memory is needed to perform computations because not all data points and kernel weights were engaged in a forward pass, but those savings pale in comparison to how much RAM is still required to store weights and intermediate results, and the training process is still as delicate as before.

2. No guaranteed hierarchical learning via SSCs alone.
One of the ideas used in the famous VGGNet [[6]](#6) that paved the way for substantially deeper networks is using a stack of `3x3` convolutions to achieve the same receptive field as a larger single kernel but at a smaller cost. For example, three consecutive `3x3` convolutions have the same receptive field as a `7x7` filter. Utilizing these minimal kernels to pass contextual information step-by-step with non-linear transformations at each level instead of doing it all at once is now a central idea of most CNNs. 

    This progressive information flow relies on the fact that **each** `3x3` convolution gathers the most relevant information available to it and passes it to the next stage. This hierarchy is essential for modern CNNs and, unfortunately, SSCs cannot guarantee it anymore. Contrary to dense convolutions, SSCs do not have a universal minimal filter that always passes information available to it. If the central cell is empty, no information will be acquired from the surroundings and this patch of input data will not be seen by the next layer.

    Here, the top left corner never gets the information about the rest of the cells since the surrounding 8 pixels always remain inactive per SSC's design. If a single SSC kernel was used to build a donut/not donut classifier, it would fail miserably by never activating.

    <p align="center">
      <img src="https://raw.githubusercontent.com/aleksandrkim/blog/master/attachments/minimal_filter.png">
    </p>
    
    The only way hierarchical learning is possible using only the smallest SSCs is if all active sites are connected. The cell with `2` inside acts as a bridge between the central cell `1` and the others.

    <p align="center">
      <img src="https://raw.githubusercontent.com/aleksandrkim/blog/master/attachments/minimal_filter_chain.png">
    </p>

    Strided SSCs could be used to gradually bring all active sites closer together through downsampling, but this comes at the cost of translational invariance since a small translation might deactivate a previously active kernel.

    <p align="center">
      <img src="https://raw.githubusercontent.com/aleksandrkim/blog/master/attachments/strided_ssc.png">
    </p>

    Another way to enable hierarchical learning is by using SC-based operations: SC or pooling. This solves the problem, but also means that SSCs are only applicable in some scenarios and are not universal layers like dense convolutions. 

    If you look at the architectures that were evaluated to demonstrate the power of SSCs, you will notice that the authors had to use regular convolutions whenever the spatial resolution was changing since SSCs are only reliable with `stride=1`.

    This plot compares three different architectures evaluated on ShapeNet, where `C3` is a model made of only SSCs and, thus, only operates on a single spatial resolution. Evidently, SSCs alone are not enough to produce good results (only marginally better than dense CNNs) and should be used only in conjunction with other operators. In other words, SSCs are not as versatile as standard convolutions, which can be adjusted with different stride, dilation, and size values to fulfill different roles depending on the context. 

    <p align="center">
      <img src="https://raw.githubusercontent.com/aleksandrkim/blog/master/attachments/c3.png">
    </p>
    
    
    
### Conclusion
To summarize, the paper introduced a new convolutional filter designed to process sparse data and built an open-source implementation of efficient and accurate 3D CNNs for semantic segmentation based on this new idea. The submanifold sparse convolutional operator is a good alternative to standard dense convolutions but is not as universally applicable. Read the original work to find out more details and check out the associated [github repository](https://github.com/facebookresearch/SparseConvNet) for the code.

[The spotlight presentation by Ben Graham at CVPR2018 ](ttps://youtu.be/op9IBox_TTc?t=4984)



## References

<a id="1">[1]</a>
G. Riegler, A. O. Ulusoy, and A. Geiger. Octnet: Learning deep 3d representations at high resolutions. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017.

<a id="2">[2]</a> 
M. Engelcke, D. Rao, D. Z. Wang, C. H. Tong, and I. Posner. Vote3Deep: Fast Object Detection in 3D Point Clouds using Efficient Convolutional Neural Networks. IEEE International Conference on Robotics and Automation, 2017.

<a id="3">[3]</a> 
B. Graham. Sparse 3D Convolutional Neural Networks. British Machine Vision Conference, 2015.

<a id="4">[4]</a> 
A.X. Chang, T. Funkhouser, L. Guibas, P. Hanrahan, Q. Huang, Z. Li, S. Savarese, M. Savva, S. Song, H. Su, J. Xiao, L. Yi, F. Yu. ShapeNet: An Information-Rich 3D Model Repository. Tech. Rep. arXiv:1512.03012 [cs.GR], Stanford University — Princeton University — Toyota Technological Institute at Chicago (2015).

<a id="5">[5]</a> 
A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. Nießner. Scannet:  Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

<a id="6">[6]</a> 
K. Simonyan and A. Zisserman.  Very deep convolutional networks for large-scale image recognition. ICLR, 2015.

---

This blog is originally written for the seminar class "Recent Trends in 3D Computer Vision and Deep Learning" at the Technical University of Munich

