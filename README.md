# graphical-perception-with-cnn

# Data
Our data is available on [Google Drive](https://drive.google.com/drive/folders/18w26-rn-8mLp7GxLPHGdiSvfb7vc61K-?usp=sharing)
## Attributes
- **Type**: Visulization type of the bar chart as described in our paper, possible values are `1`, `2`, `3`, `4`, `5`.
- **Method**: Sampling method, possible values are `IID`, `COV`, `ADV`, `MIN`.
- **Downsampling**: Downsampling level, the number of training samples will be 1/n of the baseline, possible values are `1`, `2`, `4`, `8`, `16`.
- **LargeHeight**: Height of the taller marked bar.
- **SmallHeight**: Height of the shorter marked bar.
- **GroundTruth**: Ground truth value for training. In ratio estimation task it's the ratio of two marked bars (SmallHeight/LargeHeight), in cell counting and node counting task it's the number of cells/nodes in the image.
- **Prediction**: The prediction of our CNN models
