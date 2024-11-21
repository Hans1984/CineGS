# [CinematicGS](https://cinegs.mpi-inf.mpg.de/): Cinematic Gaussians:  Real-Time HDR Radiance Fields with Depth of Field [Pacific Graphics 2024]


### Installation
Use the following commands with Anaconda to create and activate your environment:
  - ```conda env create -f environment.yaml```
  - ```conda activate cinegs```
    
Note:
Our installation method is the same as that of [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting/tree/main). If you encounter installation issues, please refer to the issues section of it.

### Training
For real dataset:

```
 python train.py -s data_path -m output_path --fd_path .. --ap_path .. --exp_path  .. --length_focal .. --blur
```
For rendering dataset:

```
python train.py -s data_path -m output_path --length_focal .. --start_checkpoint pretrained_path/chkpnt7000.pth --blur
```


please change the base_path to your own dataset path

### Rendering
For real dataset:

```
python render.py -m model_path -s data_path --fd_path .. --ap_path .. --exp_path .. --length_focal ..
```

For rendering dataset:

```
python render.py -m model_path -s data_path
```

The --blur flag should be disabled during rendering.

### Post-Editing

```
python render_taf.py -m model_path -s data_path --fd_path .. --ap_path .. --exp_path  .. --length_focal .. --taf
```

The aperture size, focus distance, and exposure can be adjusted post-training. You can set appropriate parameters to view the editing results. (The default parameters are for the real dataset scene_3.)


<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@article{wang2024cinematic,
  title={Cinematic Gaussians: Real-Time HDR Radiance Fields with Depth},
  author={Wang, Chao and Wolski, Krzysztof and Kerbl, Bernhard and Serrano, Ana 
  and Bemana, Mojtaba and Seidel, Hans-Peter and Myszkowski, Karol and Leimk{\"u}hler, Thomas},
  booktitle={Computer Graphics Forum},
  volume={43},
  number={7},
  pages={1--13},
  year={2024},
  organization={Blackwell-Wiley}
}
}</code></pre>
  </div>
</section>

### Acknowledge
This source code is derived from the (https://github.com/graphdeco-inria/gaussian-splatting/tree/main), and the depth rendering part is referred to (https://github.com/leo-frank/diff-gaussian-rasterization-depth). We really appreciate the contributions of the authors to that repository.



