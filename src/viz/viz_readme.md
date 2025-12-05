### Readme to explain how to create visualizations for both Maskformer and Unet test forward passes

You can call:

`python visualization.py`

`--image-ind` argument is optional, if not specified, it will choose a random image in the test dataloader

`--run` argument is required, and is expected to be in the same format as expected by the flood_maskformer.load_weights method, i.e. "maskformer_run_timestamp", e.g. "maskformer_run_20251204_02:04"

```python

```
