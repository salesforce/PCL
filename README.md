## PCL: Prototypical Contrastive Learning of Unsupervised Representations (Salesforce Research)
<img src="./img/PCL_framework.png" width="600">

This is a PyTorch implementation of the PCL paper:
<pre>
@article{PCL,
	title={Prototypical Contrastive Learning of Unsupervised Representations},
	author={Junnan Li and Pan Zhou and Caiming Xiong and Richard Socher and Steven C.H. Hoi},
	journal={arXiv preprint arXiv:2005.04966},
	year={2020}
}</pre>

### Requirements:
* ImageNet dataset
* Python ≥ 3.6
* PyTorch ≥ 1.4
* <a href="https://github.com/facebookresearch/faiss">faiss-gpu</a>: pip install faiss-gpu
* pip install tqdm

### Unsupervised Training:
Similar as <a href="https://github.com/facebookresearch/moco">MoCo</a>, this implementation only supports multi-gpu, DistributedDataParallel training, which is faster and simpler; single-gpu or DataParallel training is not supported.

To perform unsupervised training of a ResNet-50 model on ImageNet, run: \
<code>python main_pcl.py \
  -a resnet50 \. 
  --lr 0.03 \
  --batch-size 256 \
  --mlp --cos --aug-plus\	
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --exp-dir PCL_v2\
</code>\


