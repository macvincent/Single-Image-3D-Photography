# Sampling Novel Views from a Single 2D Image

Given an input RGB image, we simulate a 3D interactive experience by generating novel viewpoints to account for disocclusion. Builds upon SLIDE: Single Image 3D Photography with Soft Layering and Depth-aware Inpainting by Jamapani et al., ICCV 2021 (Oral). Unlike SLIDE, we outpaint the input image with a denoising probabilistic diffusion model, use a matting model to separate the background from the foreground image, and construct our meshes with the Open3D library in place of Tensorflow 3D.


![sample_output](https://github.com/macvincent/Sampling-Novel-Views-from-a-Single-2D-Image/blob/master/outputs/class.gif)

# Getting Started
* Setup and activate conda environment.
```bash
conda create -n sample_novel_views python=3.8
conda activate sample_novel_views
```
* Install required pre-requisites.
```bash
pip install -r requirements.txt
```
* Add additional `.jpg` or `.png` test images to the `images` folder.
* Run the the code using this command
```bash
python main.py --config config.yaml
```

For each image in the `images` folder the relevant meshes will be saved to the `meshes` folder and the output videos will be saved to the `outputs` folder.

# Acknowledgments

Our work builds upon `SLIDE: Single Image 3D Photography with Soft Layering and Depth-aware Inpainting` by [Jamapani et al. ICCV 2021 (Oral)](https://varunjampani.github.io/slide/). For code structure, we drew inspiration from `3D Photography using Context-aware Layered Depth Inpainting` by [Shih et al. CVPR 2020](https://github.com/vt-vl-lab/3d-photo-inpainting/blob/master/README.md). Thank you to the authors of the latter paper for making their code publicly available.
