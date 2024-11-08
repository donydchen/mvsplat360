<p align="center">
  <h1 align="center">MVSplat360: Feed-Forward 360 Scene Synthesis <br> from Sparse Views</h1>
  <p align="center">
    <a href="https://donydchen.github.io/">Yuedong Chen</a>
    &nbsp;·&nbsp;
    <a href="https://chuanxiaz.com/">Chuanxia Zheng</a>
    &nbsp;·&nbsp;
    <a href="https://haofeixu.github.io/">Haofei Xu</a>
    &nbsp;·&nbsp;
    <a href="https://bohanzhuang.github.io/">Bohan Zhuang</a> <br>
    <a href="https://www.robots.ox.ac.uk/~vedaldi/">Andrea Vedaldi</a>
    &nbsp;·&nbsp;
    <a href="https://personal.ntu.edu.sg/astjcham/">Tat-Jen Cham</a>
    &nbsp;·&nbsp;
    <a href="https://jianfei-cai.github.io/">Jianfei Cai</a>
  </p>
  <h3 align="center">NeurIPS 2024</h3>
  <h3 align="center"><a href="https://arxiv.org/abs/2411.04924">Paper</a> | <a href="https://donydchen.github.io/mvsplat360/">Project Page</a> | <a href="https://huggingface.co/donydchen/mvsplat360/tree/main">Pretrained Models</a> </h3>

<br>
</p>

https://github.com/user-attachments/assets/4cfa6654-5bb5-4f72-a264-6941bcf00bed

## Installation

To get started, create a conda virtual environment using Python 3.10+ and install the requirements:

```bash
conda create -n mvsplat360 python=3.10
conda avtivate mvsplat360
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 xformers==0.0.25.post1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Acquiring Datasets

This project mainly uses [DL3DV](https://github.com/DL3DV-10K/Dataset) and [RealEstate10K](https://google.github.io/realestate10k/index.html) datasets.

The dataset structure aligns with our previous work, [MVSplat](https://github.com/donydchen/mvsplat?tab=readme-ov-file#acquiring-datasets). You may refer to the script [convert_dl3dv.py](src/scripts/convert_dl3dv.py) for converting the DL3DV-10K datasets to the torch chunks used in this project.

You might also want to check out the [DepthSplat's DATASETS.md](https://github.com/cvg/depthsplat/blob/main/DATASETS.md), which provides detailed instructions on pre-processing DL3DV and RealEstate10K for use here (as both projects share the same code base from pixelSplat).

A pre-processed tiny subset of DL3DV (containing 5 scenes) is provided [here](https://huggingface.co/donydchen/mvsplat360/blob/main/dl3dv_tiny.zip) for quick reference. To use it, simply download it and unzip it to `datasets/dl3dv_tiny`.


## Running the Code

### Evaluation

To render novel views,

* get the pretrained models [dl3dv_480p.ckpt](https://huggingface.co/donydchen/mvsplat360/blob/main/dl3dv_480p.ckpt), and save them to `/checkpoints`

* run the following:

```bash
# dl3dv; requires at least 22G VRAM
python -m src.main +experiment=dl3dv_mvsplat360 \
wandb.name=dl3dv_480P_ctx5_tgt56 \
mode=test \
dataset/view_sampler=evaluation \
dataset.roots=[datasets/dl3dv_tiny] \
checkpointing.load=outputs/dl3dv_480p.ckpt
```

* the rendered novel views will be stored under `outputs/test/{wandb.name}`

To evaluate the quantitative performance, kindly refer to [compute_dl3dv_metrics.py](src/scripts/compute_dl3dv_metrics.py)

To render videos from a pretrained model, run the following

```bash
# dl3dv; requires at least 38G VRAM
python -m src.main +experiment=dl3dv_mvsplat360_video \
wandb.name=dl3dv_480P_ctx5_tgt56_video \
mode=test \
dataset/view_sampler=evaluation \
dataset.roots=[datasets/dl3dv_tiny] \
checkpointing.load=outputs/dl3dv_480p.ckpt 
```

### Training

* Download the encoder pretrained weight from [MVSplat](https://github.com/donydchen/mvsplat) and save it to `checkpoints/re10k.ckpt`.
* Download SVD pretrained weight from [generative-models](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/tree/main) and save it to `checkpoints/svd.safetensors`
* Run the following:

```bash
# train mvsplat360; requires at least 80G VRAM
python -m src.main +experiment=dl3dv_mvsplat360 dataset.roots=[datasets/dl3dv]
```

* To fine tune from our released model, append `checkpointing.load=outputs/dl3dv_480p.ckpt` and `checkpointing.resume=false` to the above command. 
* You can also set up your wandb account [here](config/main.yaml) for logging. Have fun.

## BibTeX

```bibtex
@article{chen2024mvsplat360,
    title     = {MVSplat360: Feed-Forward 360 Scene Synthesis from Sparse Views},
    author    = {Chen, Yuedong and Zheng, Chuanxia and Xu, Haofei and Zhuang, Bohan and Vedaldi, Andrea and Cham, Tat-Jen and Cai, Jianfei},
    booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
    year      = {2024},
}
```

## Acknowledgements

The project is based on [MVSplat](https://github.com/donydchen/mvsplat), [pixelSplat](https://github.com/dcharatan/pixelsplat), [UniMatch](https://github.com/autonomousvision/unimatch) and [generative-models](https://github.com/Stability-AI/generative-models). Many thanks to these projects for their excellent contributions!
