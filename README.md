# Multi-Graph Transformer for Free-Hand Sketch Recognition

<div align=center><img src="https://github.com/PengBoXiangShang/multigraph_transformer/blob/master/figures/cat.gif" width = 20% height = 20% /></div>

<div align=center><img src="https://github.com/PengBoXiangShang/multigraph_transformer/blob/master/figures/cat_graph.png"/></div>


This code repository is the official source code of the paper "Multi-Graph Transformer for Free-Hand Sketch Recognition" ([ArXiv Link](https://github.com/PengBoXiangShang/multigraph_transformer)).


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
We also provide our experimental results including trainging log files, model checkpoints, and TensorBoard curves. The following table provides the download links in Google Drive, which is corresponding to the Table 3 in our paper. **“MGT #15”** is the full model of our MGT.

Network | training script | acc. | log & ckpts & TensorBoard curves
:-: | :-: | :-: | :-:
GT #1 | [train_gra_transf_inpt5_new_dropout_2layerMLP_fully_connected_graph_early_stop.py](https://github.com/PengBoXiangShang/multigraph_transformer/blob/master/train_gra_transf_inpt5_new_dropout_2layerMLP_fully_connected_graph_early_stop.py) | 0.5249 | [link](https://drive.google.com/open?id=18F4-K8MdjL5cTtDAkMTGRtSkvoTpsNlu), 50M, MD5 checksum 1f703a7aeb38a981bb430965a522b33a.
GT #2 | [train_gra_transf_inpt5_new_dropout_2layerMLP_fully_connected_stroke_early_stop.py](https://github.com/PengBoXiangShang/multigraph_transformer/blob/master/train_gra_transf_inpt5_new_dropout_2layerMLP_fully_connected_stroke_early_stop.py)  |0.6487 | [link](https://drive.google.com/open?id=1sQA-gDaH5N1AQu3lCkD-AIeapzC4_F05), 50M, MD5 checksum 0428b6a80e84413ba3e15cd591325668.
GT #3 | [train_gra_transf_inpt5_new_dropout_2layerMLP_random_attention_mask_early_stop.py](https://github.com/PengBoXiangShang/multigraph_transformer/blob/master/train_gra_transf_inpt5_new_dropout_2layerMLP_random_attention_mask_early_stop.py) |0.5271 | [link](https://drive.google.com/open?id=1gk91Gn8jGNZuV4OSohhEo9zqwDkpyAxL), 50M, MD5 checksum 7a3724c2b9926187b1a9c1c48e246a11.
GT #4 | [train_gra_transf_inpt5_new_dropout_2layerMLP_random_attention_mask_early_stop_20percent.py](https://github.com/PengBoXiangShang/multigraph_transformer/blob/master/train_gra_transf_inpt5_new_dropout_2layerMLP_random_attention_mask_early_stop_20percent.py) | 0.5352| [link](https://drive.google.com/open?id=1gwJXIT_lh6S0isT_4K44HHz7QXvlYkCN), 50M, MD5 checksum e27797f991720551bca26b9f87d99fac.
GT #5 | [train_gra_transf_inpt5_new_dropout_2layerMLP_random_attention_mask_early_stop_30percent.py](https://github.com/PengBoXiangShang/multigraph_transformer/blob/master/train_gra_transf_inpt5_new_dropout_2layerMLP_random_attention_mask_early_stop_30percent.py) | 0.5322| [link](https://drive.google.com/open?id=1E3n4VjEBOK88OJ5ud1d7OpeqU41khrEv), 50M, MD5 checksum 1ba1939a6266425c8ac843fe06a8b8b9.
GT #6 | [train_gra_transf_inpt5_new_dropout_2layerMLP_2nn_early_stop.py](https://github.com/PengBoXiangShang/multigraph_transformer/blob/master/train_gra_transf_inpt5_new_dropout_2layerMLP_2nn_early_stop.py) | 0.7023| [link](https://drive.google.com/open?id=1bKz1XtIGpuhYxH0xPUNOy1C__P5T4ccK), 50M, MD5 checksum bca3bd8052d1fbc01797ac688c5061d4.
GT #7 | [train_gra_transf_inpt5_new_dropout_2layerMLP_4nn_early_stop.py](https://github.com/PengBoXiangShang/multigraph_transformer/blob/master/train_gra_transf_inpt5_new_dropout_2layerMLP_4nn_early_stop.py) |0.7082 | [link](https://drive.google.com/open?id=1MyWrxdVZNbpYrxTCq4Vf5BlgjWZaNrfp), 50M, MD5 checksum 8615fd91d5291380b9c027ad6dd195d8.
GT #8 | [train_gra_transf_inpt5_new_dropout_2layerMLP_6nn_early_stop.py](https://github.com/PengBoXiangShang/multigraph_transformer/blob/master/train_gra_transf_inpt5_new_dropout_2layerMLP_6nn_early_stop.py) |0.7028 | [link](https://drive.google.com/open?id=1dk30oUOmRGyXOsuTJMhGCP__uMbaaJCD), 50M, MD5 checksum 36c6126ad9a05ad7b55e7e76c175243e.
GT #9 | [train_gra_transf_inpt5_new_dropout_2layerMLP_jnn_early_stop.py](https://github.com/PengBoXiangShang/multigraph_transformer/blob/master/train_gra_transf_inpt5_new_dropout_2layerMLP_jnn_early_stop.py) |0.5488| [link](https://drive.google.com/open?id=1jS9R3-9mAIageGLGdEc_gUHs_Dy_3dpq), 50M, MD5 checksum ef59dfe9abe3c71487201e886832a559.
MGT #10 | [train_gra_transf_inpt5_new_dropout_2layerMLP_2nn4nn_early_stop.py](https://github.com/PengBoXiangShang/multigraph_transformer/blob/master/train_gra_transf_inpt5_new_dropout_2layerMLP_2nn4nn_early_stop.py) | 0.7149 | [link](https://drive.google.com/open?id=1uy68jczYC-GHM-mWGb4EGMC_zj4nv8LC), 100M, MD5 checksum c7868b466e5946b0ebfc5ba52bb83b94.
MGT #11 | [train_gra_transf_inpt5_new_dropout_2layerMLP_2nnjnn_early_stop.py](https://github.com/PengBoXiangShang/multigraph_transformer/blob/master/train_gra_transf_inpt5_new_dropout_2layerMLP_2nnjnn_early_stop.py) | 0.7111 | [link](https://drive.google.com/open?id=1lcxQOtCIMHwy3E6Jr0hAcomXEJnRogSU), 100M, MD5 checksum d91b95915db89cb73645cd8cbe2ad139.
MGT #12 | [train_gra_transf_inpt5_new_dropout_2layerMLP_4nnjnn_early_stop.py](https://github.com/PengBoXiangShang/multigraph_transformer/blob/master/train_gra_transf_inpt5_new_dropout_2layerMLP_4nnjnn_early_stop.py) | 0.7237| [link](https://drive.google.com/open?id=1cffhA2O8t8JyGd-824xMw2dpTbM3Ve1T), 100M, MD5 checksum 12958648e3c392bf62d96ec30cf26b79.
MGT #13 | [train_gra_transf_inpt5_new_dropout_2layerMLP_2nn2nn2nn_early_stop.py](https://github.com/PengBoXiangShang/multigraph_transformer/blob/master/train_gra_transf_inpt5_new_dropout_2layerMLP_2nn2nn2nn_early_stop.py) | 0.7077 | [link](https://drive.google.com/open?id=13pd2MraOEUMILLwZtVPAjEsGye_W219A), 141M, MD5 checksum 88c417afa5710bffea06e89b6029dae7.
MGT #14 | [train_gra_transf_inpt5_new_dropout_2layerMLP_2nn4nn6nn_early_stop.py](https://github.com/PengBoXiangShang/multigraph_transformer/blob/master/train_gra_transf_inpt5_new_dropout_2layerMLP_2nn4nn6nn_early_stop.py) | 0.7156| [link](https://drive.google.com/open?id=1Ga0Mh897JQksiiqy77BGh2klfybi5oS4), 141M, MD5 checksum c5ccb84f347a9332505afd8728d915fe.
**MGT #15** | [train_gra_transf_inpt5_new_dropout_2layerMLP_2nn4nnjnn_early_stop.py](https://github.com/PengBoXiangShang/multigraph_transformer/blob/master/train_gra_transf_inpt5_new_dropout_2layerMLP_2nn4nnjnn_early_stop.py) | **0.7280**| [link](https://drive.google.com/open?id=1qEGk84k8KGK93jRD9OIlW1Ed4c5Iq96Z), 141M, MD5 checksum 7afe439e34f55eb64aa7463134d67367.

## Citations
If you find this code useful to your research, please cite our paper as the following bibtex:
ArXiv ...

## Acknowledgement
Many thanks to the great sketch dataset [**Quick, Draw！**](https://github.com/googlecreativelab/quickdraw-dataset) released by Google.

## FAQ
Please see FAQ via this [link](https://github.com/PengBoXiangShang/multigraph_transformer).
