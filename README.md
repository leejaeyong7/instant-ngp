# Instant-PPNG
This repository contains the customized Instant-NGP codebased for Plenoptic PNG: Real-Time Neural Radiance Fields in 150 KB
PPNG allows real-time rendering of NeRF models in a web-browser, without any post-processing process (e.g., baking).
PPNG also trains roughly on the same speed of HashGrid, using similar file size. 

Please refer to the technical paper for the details.

## Installation
Please follow the original installation instructions for the Instant-NGP repository. 
One major modification from the original repository is the use of modified [tiny-cuda-nn](https://github.com/leejaeyong7/tiny-cuda-nn) repository which contains the PPNG implementation.
Since the submodules are set using the correct commit, simply installing the NGP repository should suffice. 

## Running
To train the model with the PPNG, please use:
```bash
python scripts/run_ppng.py --run_name fox \
                           --scene_path data/nerf/fox/transforms.json \
                           --output_path outputs/

# OR, to run with test file supports
python scripts/run_ppng_with_test.py --run_name fox \
                                     --scene_path ../data/chair/transforms_train.json \
                                     --test_scene_path ../data/chair/transforms_test.json \
                                     --output_path outputs/
```
This will train PPNG on data provided by `scene_path`, to output `output_path/fox.ingp` and `output_path/fox.ppng` files.
`output_path/fox.ingp` contains Instant-NGP weights and `output_path/fox.ppng` contains translated file that can be rendered via web-browser. 

We additionally support interactive run with:
```bash
./instant-ngp --config configs/nerf/ppng_2.json data/nerf/fox
```

and python headless run with various config with:
```bash
python scripts/run.py --config configs/nerf/ppng_2.json --scene data/nerf/fox --save_snapshot PATH_TO_SAVE_INGP_FILE
```

## Visualizing the runs

Files with PPNG extensions can be visualized by our interactive viewer available at [separate](https://github.com/leejaeyong7/ppng) repository.

We additionally provide a code to translate trained `ingp` files of PPNG weights into a render friendly format. 
```bash
python scripts/bake.py --ingp_file PATH_TO_SAVED_INGP_FILE.ingp --output_file PATH_TO_OUTPUT_FILES.ppng
```


## Original License and Citation
Please cite the original Instant-NGP paper and the PPNG for use of this work

```bibtex
@article{mueller2022instant,
    author = {Thomas M\"uller and Alex Evans and Christoph Schied and Alexander Keller},
    title = {Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
    journal = {ACM Trans. Graph.},
    issue_date = {July 2022},
    volume = {41},
    number = {4},
    month = jul,
    year = {2022},
    pages = {102:1--102:15},
    articleno = {102},
    numpages = {15},
    url = {https://doi.org/10.1145/3528223.3530127},
    doi = {10.1145/3528223.3530127},
    publisher = {ACM},
    address = {New York, NY, USA},
}
```

Copyright © 2022, NVIDIA Corporation. All rights reserved.
This work is made available under the Nvidia Source Code License-NC. Click [here](LICENSE.txt) to view a copy of this license.
