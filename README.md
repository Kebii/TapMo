# TapMo: Shape-aware Motion Generation of Skeleton-free Characters

This is the code for [TapMo: Shape-aware Motion Generation of Skeleton-free Characters](https://arxiv.org/abs/2310.12678) by Jiaxu Zhang, et al.


![](https://github.com/Kebii/Tapmo/blob/main/gifs/teaser.jpg)
TapMo is a text-based animation pipeline for generating motion in a wide variety of skeleton-free characters.

![](https://github.com/Kebii/Tapmo/blob/main/gifs/demo1.gif)
![](https://github.com/Kebii/Tapmo/blob/main/gifs/demo2.gif)


An overview of the TapMo pipeline. Given a non-rigged mesh and a motion description input by the user, the Mesh Handle Predictor $\lambda (\cdot)$ predicts mesh handles and skinning weights to control the mesh. The Shape-aware Motion Diffusion $\mu (\cdot)$ generates a text-guided and mesh-specific motion for the character using the motion description and the mesh deformation feature ${f}_{\phi}$ extracted by the Mesh Handle Predictor.
![](https://github.com/Kebii/Tapmo/blob/main/gifs/method.jpg)
