# MCT-ARG
We propose an innovative multi-channel Transformer-based framework for ARG prediction, termed MCT-ARG. By integrating protein sequence data, secondary structure information, and relative solvent accessibility (RSA) features, our approach overcomes the traditional reliance on single-sequence features, markedly improving both the accuracy of ARG identification and the resolution of functional site analysis. The model employs a multi-head self-attention mechanism to capture global dependencies among multimodal features and incorporates a dual-constraint regularization strategy—comprising entropy minimization and local continuity constraints—to optimize the attention weight distribution and strengthen the detection of key functional sites.

<br>
MCT-ARG is based on the pytorch framework. Please configure the pytorch environment before using it.
<br>

If you want to predict, please prepare the protein sequence file, the corresponding secondary structure file, and the RSA file of the sequence to be predicted, and then enter the following command in the terminal to get the result. You can use SCRATCH-1D to obtain protein secondary structure and RSA information.
```python
python predict.py -a  test.fasta -s  test.ss8 -r test.acc20 -o output
```
<br>
If you want to use MCT-ARG to train your own dataset, please prepare the protein sequence file, the corresponding secondary structure file and the RSA file first, and then enter the following command in the terminal to get the trained model. The trained model will be saved in the ./model directory file.
```bash
python train.py -a  input.fasta -s  input.ss8 -r input.acc20
```

