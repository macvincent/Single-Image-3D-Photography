# [Final Project CS231A] Sampling Novel Views from a Single 2D Image

Imagine walking into a lecture hall. As you walk along, more objects become visible to you and others slowly recede from your field of view. Now, imagine a different scenario. This time, we are given a picture of the lecture hall. A few seats are captured within the picture frame but most of the lecture hall is left off this image. Even without a complete picture of the room, we can imagine what we would see if we physically walked into the lecture hall. From that single picture, we have an expectation of how the image of the room would change as we focus on different portions of the scene. In this project, we simulate a similar experience. Given an input 2D RGB image, we will generate novel viewpoints that simulate a 3D interactive experience.


![sample_output](https://github.com/macvincent/Sampling-Novel-Views-from-a-Single-2D-Image/blob/master/outputs/horse.gif)

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
