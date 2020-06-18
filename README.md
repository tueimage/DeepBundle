# Improving Performance of the DeepBundle Tractography Parcellation
This repository contains code that emulates the DeepBundle [[1]](#1) framework, visualizes its features, and explores the addition of an SVM on top of the network and false positive mining to improve performance.

## Setup
A publicly available dataset [[2]](#2) containing 105 subjects (72 bundles each) from the Human Connectome Project (HCP) [[3]](#3) is used.
Python 3 with Tensorflow 1.15.0 are the primary requirements, the `req.txt` file contains a listing of other dependencies. 
To install all the requirements, run the following:
```
$ pip install -r req.txt
```

Model parameters and directories for loading data and saving results are defined in the `params.py` file, the code can be run using:
```
$ python main.py
```

## References
<a id="1">[1]</a> 
F. Liu, J. Feng, G. Chen, Y. Wu, Y. Hong, P. T. Yap, and D. Shen,
[DeepBundle: Fiber Bundle Parcellation with Graph Convolution NeuralNetworks](https://arxiv.org/abs/1906.03051), 
in Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics),
vol. 11849 LNCS, 2019, pp. 88–95

<a id="2">[2]</a> 
J. Wasserthal, P. F. Neher, and K. H. Maier-Hein, [Tract orientation mapping for bundle-specific tractography](https://link.springer.com/chapter/10.1007/978-3-030-00931-1_5), 
in Lecture Notes in ComputerScience (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 
vol. 11072 LNCS.Springer Verlag,9 2018, pp. 36–44.

<a id="3">[3]</a>
D. C. Van Essen, S. M. Smith, D. M. Barch, T. E. Behrens, E. Yacoub, and K. Ugurbil, [The WU-Minn Human Connectome Project: An overview](https://www.sciencedirect.com/science/article/pii/S1053811913005351?casa_token=I2G5X-pQVV4AAAAA:-5zH32bEiZ-IjaxPAMZ-hAESb9L3wFlLkDNHVqMCK2LC7sAMLLDcjmi75hbsrOsBac_zIpxON8s), 
NeuroImage, vol. 80, pp. 62–79, 10 2013
