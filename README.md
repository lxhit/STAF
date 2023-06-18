# MASTAF: A Model-Agnostic Spatio-Temporal Attention Fusion Network for Few-shot Video Classification

We propose MASTAF, a Model-Agnostic Spatio-Temporal Attention Fusion network for few-shot video classification. MASTAF takes input from a general video spatial and temporal representation,e.g., using  2D CNN, 3D CNN, and video Transformer. Then, to make the most of such representations, we use self- and cross-attention models to highlight the critical spatio-temporal region to increase the inter-class distance and decrease the intra-class distance. Last, MASTAF applies a lightweight fusion network and a nearest neighbor classifier to classify each query video. We demonstrate that MASTAF improves the state-of-the-art performance on three few-shot video classification benchmarks(UCF101, HMDB51, and  Something-Something-V2), e.g., by up to 91.6\%, 69.5\%, and 60.7\% for five-way one-shot video classification, respectively.

# Getting started

**Environment**:
1. Anaconda with python >= 3.8
2. sklearn >= 0.22.1
3. seaborn >= 0.10
4. matplotlib >=3.3.0


**Data**:

To directly view the running results in the notebooks, or if you can modify the corresponding data directory in your own dataset.

# References
This algorithm library is extended from [scikit-learn Tutorials](https://scikit-learn.org/stable/tutorial/index.html).
