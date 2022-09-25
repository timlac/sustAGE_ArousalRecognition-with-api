# The sustAGE off-the-shelf arousal recognition toolkit

This repository provides an off-the-shelf arousal recognition API, which can be used to infer whether the speaker recorded in the input speech file conveys a low, mid, or high arousal level. 

## General Installation Instructions

### Linux
If you have conda installed (either miniconda or anaconda), you can execute
```bash
conda env create -f .env-ymls/FIPsustAGE.yml
```
to setup the virtual environment required to execute the API. You can activate the `FIPsustAGE` environment with 
```bash
source ./activate FIPsustAGE
```

## Source code
`src` contains the back-end Python scripts of the API, including the generation and segmentation of the Mel-spectrogram representations of the input speech file, and their analysis with a Convolutional Neural Network. 

## Demonstration
Next, we provide an example on how to execute the provided API inside a Python program.

```python
>>> import module_arousal as arousal
>>> arousal.API('[audioFilePath].wav')
('mid', '0.524')
```

The first element corresponds to the arousal level inferred by the model (`low`, `mid`, or `high`), and the second element indicates the probability score associated to the inference. 

## Citation
If you use the code from this repository, you are kindly asked to cite the following paper.

**[TBD]**

## License
The code and the model weights are released under the MIT License.

## Acknowledgements
The research and development of this toolkit has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No. 826506 (sustAGE).
