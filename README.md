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

To perform unsupervised training of a ResNet-50 model on ImageNet using a 4-gpu or 8-gpu machine, run: 
<pre>python main_pcl.py \ 
  -a resnet50 \ 
  --lr 0.03 \
  --batch-size 256 \
  --temperature 0.2 \
  --mlp --cos --aug-plus \	
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --exp-dir PCL_v2
  [Imagenet dataset folder]
</pre>

### Linear SVM Evaluation on VOC
To train a linear SVM classifier on VOC dataset, using frozen representations from a pre-trained model, run:
<pre>python eval_svm_voc.py --pretrained [your pretrained model] \
  --low-shot (only for low-shot evaluation, otherwise the entire dataset is used)
  [VOC2007 dataset folder]
</pre>

### Linear Classifier Evaluation on ImageNet
Requirement: pip install tensorboard_logger \
To train a logistic regression classifier on ImageNet, using frozen representations from a pre-trained model, run:
<pre>python eval_cls_imagenet.py --pretrained [your pretrained model] \
  -a resnet50 \ 
  --lr 5\
  --batch-size 256 \
  --id PCL_v2_linear \ 
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [Imagenet dataset folder]
</pre>

