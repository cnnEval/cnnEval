# graphical-perception-with-cnn

# Codes
To run the experiment, use `python {script} {bar_type} {method} {run_idx} {divisor}`,
- **{bar_type}**: Visulization type of the bar chart. possible values are `1`, `2`, `3`, `4`, `5`. In our experiment we used `1` (position) and `4` (length).
- **{method}**: Sampling method, possible values are `IID`, `COV`, `ADV`, `OOD`.
- **{run_idx}**: Works as a random seed of the the experiment.
- **{divisor}**: Downsampling level of the experiment, the number of training samples will be 1/n of all training samples, possible values are `1`, `2`, `4`, `8`, `16`.

# Data
Our cell/node counting data and the results of all our experiments are available on [Google Drive](https://drive.google.com/drive/folders/1vLS8k2ZkWNOYdp1hk33f9XVnCL03jkIq?usp=sharing), saved as csv files under Formatted folder.
## Attributes
- **Type**: Visulization type of the bar chart as described in our paper, possible values are `1`, `2`, `3`, `4`, `5`.
- **Method**: Sampling method, possible values are `IID`, `COV`, `ADV`, `OOD`.
- **Downsampling**: Downsampling level, the number of training samples will be 1/n of all training samples, possible values are `1`, `2`, `4`, `8`, `16`.
- **LargeHeight**: Height of the taller marked bar.
- **SmallHeight**: Height of the shorter marked bar.
- **GroundTruth**: Ground truth value for training. In ratio estimation task it's the ratio of two marked bars (SmallHeight/LargeHeight), in cell counting and node counting task it's the number of cells/nodes in the image.
- **Prediction**: The prediction of our CNN models
