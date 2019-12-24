# Multi-Graph Transformer for Free-Hand Sketch Recognition

![](https://img.shields.io/badge/language-Python-{green}.svg)
![](https://img.shields.io/npm/l/express.svg)

<div align=center><img src="https://github.com/PengBoXiangShang/multigraph_transformer/blob/master/figures/cat.gif" width = 30% height = 30% /></div>

<div align=center><img src="https://github.com/PengBoXiangShang/multigraph_transformer/blob/master/figures/cat_graph.png"/></div>


This code repository is the official source code of the paper "Multi-Graph Transformer for Free-Hand Sketch Recognition" ([ArXiv Link](https://github.com/PengBoXiangShang/multigraph_transformer)), by [Peng Xu](http://www.pengxu.net/), [Chaitanya K. Joshi](https://chaitjo.github.io/), [Xavier Bresson](https://www.ntu.edu.sg/home/xbresson/).


<div align=center><img src="https://github.com/PengBoXiangShang/multigraph_transformer/blob/master/figures/MGT_pipeline_details.png"/></div>


## Requirements
Ubuntu 16.04.10

Anaconda 4.7.10

Python 3.7

PyTorch 1.2.0

How to install (clone) our environment will be detailed in the following.
First of all, please install Anaconda.

Our hardware environment: 2 Intel(R) Xeon(R) CPUs (E5-2690  v4  @ 2.60GHz), 128 GB RAM, 4 GTX 1080 Ti GPUs.

All the following codes can run on single GTX 1080 Ti GPU.

## Usage (How to Train Our MGT)

```
# 1. Choose your workspace and download our repository.
cd ${CUSTOMIZED_WORKSPACE}
git clone https://github.com/PengBoXiangShang/multigraph_transformer

# 2. Enter the directory.
cd multigraph_transformer

# 3. Clone our environment, and activate it.
conda-env create --name ${CUSTOMIZED_ENVIRONMENT_NAME} --file ./MGT_environment.yml
conda activate ${CUSTOMIZED_ENVIRONMENT_NAME}

# 4. Download our training/evaluation/testing datasets and the associated URL lists from our Google Drive folder. Then extract them into './dataloader' folder. data.tar.gz is 118MB, and its MD5 checksum is 8ce7347dfcc9f02376319ce321bbdd31.
cd ./dataloader
chmod +x download.sh
./download.sh
# If this script 'download.sh' can not work for you, please manually download data.tar.gz to current path via this link https://drive.google.com/open?id=1I4XKajNP6wtCpek4ZoCoVr2roGSOYBbW .
tar -zxvf data.tar.gz
rm -f data.tar.gz
cd ..

# 5. Train our MGT. Please see details in our code annotations.
# Please set the input arguments based on your case.
# When the program starts running, a folder named 'experimental_results/${CUSTOMIZED_EXPERIMENT_NAME}' will be created automatically to save your log, checkpoint, and TensorBoard curves.
python train_gra_transf_inpt5_new_dropout_2layerMLP_2nn4nnjnn_early_stop.py 
    --exp ${CUSTOMIZED_EXPERIMENT_NAME}   
    --batch_size ${CUSTOMIZED_SIZE}   
    --num_workers ${CUSTOMIZED_NUMBER} 
    --gpu ${CUSTOMIZED_GPU_NUMBER}

```

## Our Experimental Results
In order to fully demonstrate the traits of our MGT to both graph and sketch researchers,
we will provide the codes of all our ablative models reported in our paper.
We also provide our experimental results including trainging log files, model checkpoints, and TensorBoard curves. The following table provides the download links in Google Drive, which is corresponding to the Table 3 in our paper.  

"GT #1" is the original Transformer [[Vaswani *et al*.]](https://arxiv.org/abs/1706.03762), representing each input graph as a fully-connected graph.  
"GT #7" is a Transformer variant, representing each input graph as a sparse graph, *i.e.*, A^{2-hop} structure defined in our paper.  
"MGT #13" is an ablative variant of our MGT, representing each input graph as two sparse graphs, *i.e.*, A^{2-hop} and A^{global}.  
**“MGT #17”** is the full model of our MGT, representing each input graph as three sparse graphs, *i.e.*, A^{1-hop}, A^{2-hop}, and A^{global}.  
In the following table and diagram, we can see that multiple sparsely-connected graphs improve the performance of Transformer.  
Please see details in [our ArXiv paper](https://github.com/PengBoXiangShang/multigraph_transformer).

Network | acc. | log & ckpts & TensorBoard curves | training script
:-: | :-: | :-: | :-
GT #1 | 0.5249 | [link](https://drive.google.com/open?id=18F4-K8MdjL5cTtDAkMTGRtSkvoTpsNlu), 50M, MD5 checksum 1f703a7aeb38a981bb430965a522b33a. | [train_gra_transf_inpt5_new_dropout_2layerMLP_fully_connected_graph_early_stop.py](https://github.com/PengBoXiangShang/multigraph_transformer/blob/master/train_gra_transf_inpt5_new_dropout_2layerMLP_fully_connected_graph_early_stop.py) 
GT #7 | 0.7082 | [link](https://drive.google.com/open?id=1MyWrxdVZNbpYrxTCq4Vf5BlgjWZaNrfp), 50M, MD5 checksum 8615fd91d5291380b9c027ad6dd195d8. | [train_gra_transf_inpt5_new_dropout_2layerMLP_4nn_early_stop.py](https://github.com/PengBoXiangShang/multigraph_transformer/blob/master/train_gra_transf_inpt5_new_dropout_2layerMLP_4nn_early_stop.py)
MGT #13 | 0.7237| [link](https://drive.google.com/open?id=1cffhA2O8t8JyGd-824xMw2dpTbM3Ve1T), 100M, MD5 checksum 12958648e3c392bf62d96ec30cf26b79. | [train_gra_transf_inpt5_new_dropout_2layerMLP_4nnjnn_early_stop.py](https://github.com/PengBoXiangShang/multigraph_transformer/blob/master/train_gra_transf_inpt5_new_dropout_2layerMLP_4nnjnn_early_stop.py)
**MGT #17** |  **0.7280**| [link](https://drive.google.com/open?id=1qEGk84k8KGK93jRD9OIlW1Ed4c5Iq96Z), 141M, MD5 checksum 7afe439e34f55eb64aa7463134d67367. | [train_gra_transf_inpt5_new_dropout_2layerMLP_2nn4nnjnn_early_stop.py](https://github.com/PengBoXiangShang/multigraph_transformer/blob/master/train_gra_transf_inpt5_new_dropout_2layerMLP_2nn4nnjnn_early_stop.py) 

<div align=center><img src="https://github.com/PengBoXiangShang/multigraph_transformer/blob/master/figures/accuracy.gif" width = 60% height = 60% /></div>

## Citations
If you find this code useful to your research, please cite our paper as the following bibtex:
ArXiv ...**coming soon**

## License
This project is licensed under the MIT License

## Acknowledgement
Many thanks to the great sketch dataset [**Quick, Draw！**](https://github.com/googlecreativelab/quickdraw-dataset) released by Google.

## FAQ
Please see FAQ via this [link](https://github.com/PengBoXiangShang/multigraph_transformer).  
If you would have further discussion on this code repository, please feel free to send email to Peng Xu.  
Email: **peng.xu [AT] ntu.edu.sg**
