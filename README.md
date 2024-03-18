# TapMo: Shape-aware Motion Generation of Skeleton-free Characters

This is the code for [TapMo: Shape-aware Motion Generation of Skeleton-free Characters](https://arxiv.org/abs/2310.12678) by Jiaxu Zhang, et al.


![](https://github.com/Kebii/Tapmo/blob/main/gifs/teaser.jpg)
TapMo is a text-based animation pipeline for generating motion in a wide variety of skeleton-free characters.

- [x] Inference code
- [ ] Training code

<!-- An overview of the TapMo pipeline. Given a non-rigged mesh and a motion description input by the user, the Mesh Handle Predictor $\lambda (\cdot)$ predicts mesh handles and skinning weights to control the mesh. The Shape-aware Motion Diffusion $\mu (\cdot)$ generates a text-guided and mesh-specific motion for the character using the motion description and the mesh deformation feature ${f}_{\phi}$ extracted by the Mesh Handle Predictor. -->

## Prerequisites
- Python >= 3.7
- [Pytorch](https://pytorch.org/) >= 1.4
- [Pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

## Quick Start
### 1. Conda environment
```
conda create python=3.8 --name tapmo
conda activate tapmo
```

### 2. Install dependencies
* Install the packages in `requirements.txt` and install [PyTorch 2.1.0](https://pytorch.org/)
```
pip install -r requirements.txt
```

### 3. Download the datasets and the requriements
Download the processed datasets and the requriements from [Google dirve](https://drive.google.com/drive/folders/1qViyiHHSXLD7l3RU-Fp8GSpO25td3oy-?usp=sharing)
```
cd TapMo
unzip datasets.zip -d ./
unzip weights.zip -d ./
unzip deps.zip -d ./shape_diffusion
```

### 4. Run
```
cd shape_diffusion
python3 -m sample.generate_handle_motion --model_path ../weights/diffusion_model_latest.pt --arch trans_dec --emb_trans_dec False --dataset t6d_mixrig --char_feature_path ../demo/shape_features/001.npy --save_path ../demo/motion/motion_ --text_prompt "walk forward and turn right."

cd handle_predictor
python -m motion_to_mesh --ckpt_path ../weights/handle_predictor_latest.pth --motion_path ../demo/motion/motion_0.npz --tgt_mesh_path ../demo/mesh/001.obj --save_dir ../demo/results/001
```

## Citation
Please cite our paper if you use this repository:
```
@inproceedings{zhang2024tapmo,
    title = {TapMo: Shape-aware Motion Generation of Skeleton-free Characters},
    author = {Zhang, Jiaxu and Huang, Shaoli and Tu, Zhigang and Chen, Xin and Zhan, Xiaohang and Yu, Gang and Shan, Ying},
    booktitle = {The Twelfth International Conference on Learning Representations ({ICLR})},
    year = {2024},
}
```

## Credit
We borrowed part of the codes from the following projects:  

https://github.com/zycliao/skeleton-free-pose-transfer

https://github.com/GuyTevet/motion-diffusion-model
