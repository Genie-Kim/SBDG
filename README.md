## SBDG (ICIP 2021): Official Project Webpage
This repository provides the official PyTorch implementation of the following paper:
> Self-balanced Learning For Domain Generalization<br>
> Jin Kim (Yonsei Univ.), Jiyoung Lee (NAVER AI Lab), Jungin Park (Yonsei Univ.)<br>
> Dongbo Min* (Ewha Womans Univ.), Kwanghoon Sohn* (Yonsei Univ.) (*: co-corresponding author)<br>
> ICIP 2021, Accepted as Poster<br>

> Paper: [arxiv](https://arxiv.org/abs/2108.13597)<br>

> **Abstract:** 
*Domain generalization aims to learn a prediction model on multi-domain source data such that the model can generalize to a target domain with unknown statistics. Most existing approaches have been developed under the assumption that the source data is well-balanced in terms of both domain and class. However, real-world training data collected with different composition biases often exhibits severe distribution gaps for domain and class, leading to substantial performance degradation. In this paper, we propose a self-balanced domain generalization framework that adaptively learns the weights of losses to alleviate the bias caused by different distributions of the multi-domain source data. The self-balanced scheme is based on an auxiliary reweighting network that iteratively updates the weight of loss conditioned on the domain and class information by leveraging balanced meta data. Experimental results demonstrate the effectiveness of our method overwhelming state-of-the-art works for domain generalization.*<br>

<p align="center">
  <img src="imgs/fig2.pdf" />
</p>


## Acknowledgments
Our implementation is heavily derived from [DomainBed](https://github.com/facebookresearch/DomainBed) and [Meta-Weight-Net](https://github.com/xjtushujun/meta-weight-net).
Thanks to the DomainBed and Meta-Weight-Net implementations.
