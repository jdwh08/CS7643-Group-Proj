### Readme to explain how to create visualizations for both Segformer and Unet test forward passes

You can call:

`python unet_visualization.py`

`--image-ind` argument is optional, if not specified, it will choose a random image in the test dataloader

`--run` argument is required, and is expected to be in the same format as expected by the flood_maskformer.load_weights method, i.e. "maskformer_run_timestamp", e.g. "maskformer_run_20251204_02:04"

`label-type` argument is optional, if not specified, it will default to hand-labeled data

Will produce an image of unet hidden states and predictions in the directory you ran the script from with the name:

*unet_viz_image{image-ind}_{label-type}.png*

Will also visualize the kernel weights in the same directory with the name:

*kernel_viz_unet.png*


You can call:

`python segformer_visualization.py`

`--image-ind` argument is optional, if not specified, it will choose a random image in the test dataloader

`--run` argument is required, and is expected to be in the same format as expected by the flood_maskformer.load_weights method, i.e. "maskformer_run_timestamp", e.g. "maskformer_run_20251204_02:04"

`label-type` argument is optional, if not specified, it will default to hand-labeled data

Will produce an image of unet hidden states and predictions in the directory you ran the script from with the name: 

*segformer_viz_image{image-ind}_{label-type}.png*

```python

```
