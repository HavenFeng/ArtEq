## Generalizing Neural Human Fitting to Unseen Poses With Articulated SE(3) Equivariance (ICCV2023-Oral)

\[[Project Page](https://arteq.is.tue.mpg.de/)\]
\[[arXiv](https://arxiv.org/abs/2304.10528)\]

![Teaser](https://arteq.is.tue.mpg.de/media/upload/artieq_teaser2.png)

This is the official Pytorch implementation of ArtEq. 

*ArtEq* (pron: Artique) is a carefully designed and principled method that extends SE(3) equivariance to articulated structures, enabling the direct regression of SMPL parameters from a 3D point cloud, which 
* has significant zero-shot pose generalization (45~60% better in V2V & MPJPE), 
* is 1000x faster during inference time compared to competing methods, and 
* has 97.3% fewer parameters than the SOTA.

Please refer to the [arXiv paper](https://arxiv.org/abs/2304.10528) for more details.

## Table of Contents

- [License](#license)
- [Description](#description)
  - [Setup](#setup)
  - [Training](#training)
  - [Eval](#eval)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## License

Software Copyright License for **non-commercial scientific research purposes**.
Please read carefully the [terms and conditions](https://github.com/HavenFeng/ArtEq/blob/main/LICENSE) and any accompanying documentation before you download and/or use the ArtEq model, data and software, (the "Model & Software"), including 3D meshes, blend weights, blend shapes, textures, software, scripts, and animations. By downloading and/or using the Model & Software (including downloading, cloning, installing, and any other use of this github repository), you acknowledge that you have read these terms and conditions, understand them, and agree to be bound by them. If you do not agree with these terms and conditions, you must not download and/or use the Model & Software. Any infringement of the terms of this agreement will automatically terminate your rights under this [License](./LICENSE).

## Description

This repository contains the training code used for the experiments in [Generalizing Neural Human Fitting to Unseen Poses With Articulated SE(3) Equivariance](https://arteq.is.tue.mpg.de/).

### Setup

1. Create an account at https://arteq.is.tue.mpg.de/
2. Run `./install.sh`
3. Activate the environment `conda activate arteq`

### Training

Run the following command to execute the code:

```Shell
python src/train.py \
    --EPN_input_radius 0.4 \
    --EPN_layer_num 2 \
    --aug_type so3 \
    --batch_size 2 \
    --epochs 15 \
    --gt_part_seg auto \
    --i 0 \
    --kinematic_cond yes \
    --num_point 5000
```

### Eval

Run the following command to evaluate the model:

```Shell
python src/eval.py \
    --EPN_input_radius 0.4 \
    --EPN_layer_num 2 \
    --aug_type so3 \
    --epoch 15 \
    --gt_part_seg auto \
    --i 0 \
    --kinematic_cond yes \
    --num_point 5000
```

or with `--paper_model`.


### TODO

- [x] official repo
- [ ] pretrained models release (The paper model, trained solely on the DFAUST train set of AMASS with fixed gaussian noise)
- [ ] ArtEq-XL model release (A much bigger ArtEq trained on AMASS + real world data, stay tuned!)

## Citation

If you find this Model & Software useful in your research we would kindly ask you to cite:

```
@misc{feng2023generalizing,
      title={Generalizing Neural Human Fitting to Unseen Poses With Articulated SE(3) Equivariance},
      author={Haiwen Feng and Peter Kulits and Shichen Liu and Michael J. Black and Victoria Abrevaya},
      year={2023},
      eprint={2304.10528},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgments

For functions or scripts that are based on external sources, we acknowledge the origin individually in each file.  
Here are some great resources we benefit:  
- [EPN](https://github.com/nintendops/EPN_PointCloud) for the point-wise SE(3) equivariance feature extraction.
- [smplx](https://github.com/vchoutas/smplx) for the SMPL body model.

We would also like to thank other recent public neural human fitting works that allow us to easily perform quantitative and qualitative comparisons :) [IPNet](https://github.com/bharat-b7/IPNet), [PTF](https://github.com/taconite/PTF), [LoopReg](https://github.com/bharat-b7/LoopReg).

This work was partly supported by the German Federal Ministry of Education and Research (BMBF): Tuebingen AI Center, FKZ: 01IS18039B

## Contact

For questions, please contact [haiwen.feng@tuebingen.mpg.de](mailto:haiwen.feng@tuebingen.mpg.de).

For commercial licensing (and all related questions for business applications), please contact [ps-licensing@tue.mpg.de](mailto:ps-licensing@tue.mpg.de).
