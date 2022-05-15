# MASTAF: A Model-Agnostic Spatio-Temporal Attention Fusion Network for Few-shot Video Classification

We propose MASTAF, a Model-Agnostic Spatio-Temporal Attention Fusion network for few-shot video classification. MASTAF takes input from a general video spatial and temporal representation,e.g., using  2D CNN, 3D CNN, and video Transformer. Then, to make the most of such representations, we use self- and cross-attention models to highlight the critical spatio-temporal region to increase the inter-class distance and decrease the intra-class distance. Last, MASTAF applies a lightweight fusion network and a nearest neighbor classifier to classify each query video. We demonstrate that MASTAF improves the state-of-the-art performance on three few-shot video classification benchmarks(UCF101, HMDB51, and  Something-Something-V2), e.g., by up to 91.6\%, 69.5\%, and 60.7\% for five-way one-shot video classification, respectively.

# Getting started

**Environment**:
1. Anaconda with python >= 3.8
2. Pytorch >= 1.8.1
3. Torchvision >= 0.9.1
4. Tensorflow >=2.2.0

**Pre-trained 3D CNNs embedding network**:

We use a 3D ResNet-50 with the weights pre-trained on the combined dataset with Kinetics-700, Moments in Time, and Start Action as our embedding network. We downloaded the weights from [here](https://github.com/kenshohara/3D-ResNets-PyTorch).

**Few-shot splits**:

We used [split](https://github.com/ffmpbgrnn/CMN) for SSv2-part, which are provided by the authors of the authors of [CMN](https://openaccess.thecvf.com/content_ECCV_2018/papers/Linchao_Zhu_Compound_Memory_Networks_ECCV_2018_paper.pdf) (Zhu and Yang, ECCV 2018). We also used the split from [OTAM](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cao_Few-Shot_Video_Classification_via_Temporal_Alignment_CVPR_2020_paper.pdf) (Cao et al. CVPR 2020) for SSv2-all, and splits from [ARN](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500511.pdf) (Zhang et al. ECCV 2020) for HMDB and UCF. For these splits files,  please refer to  ./save/video_datasets/splits/.

# Train and eval
Example: Run the following commands to train and evaluate for HMDB51 split:
```
$python -u STAF_run.py -c check_points_dir --data_folder Directory_to_load_the_splits_and_dataset --training_iterations 128000 --print_freq 1000 --query_per_class_test 1 --query_per_class 1 --shot 1 --way 5 --tasks_per_batch 64 --test_iters 6400 --num_test_tasks 10000 --dataset hmdb --split 3 -lr 0.01 --img_size 224 --seq_len 16 --start_gpu 0 --num_workers 10 --num_gpus 1 --train_num_classes 51 --pretrained_3dmodels --pretrained_path Directory_of_pretrained_3d_models --alpha 1.0 --lamda 0.5 --pre_data KMS 
```
Most of these are the default args.See paper for other hyperparams.

# References
This algorithm library is extended from [TRX](https://github.com/tobyperrett/trx) and [Cross-attention](https://github.com/blue-blue272/fewshot-CAN), which builds upon several existing publicly available code:  [CNAPs](https://github.com/cambridge-mlg/cnaps), [torch_videovision](https://github.com/hassony2/torch_videovision) and [R3D Backbone](https://github.com/kenshohara/3D-ResNets-PyTorch)
