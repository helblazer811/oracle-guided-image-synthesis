Morpho-MNIST: 'Global' dataset
==============================

This dataset corresponds to a random interleaving of plain, thinned and
thickened MNIST digits. It consists of the following files:

- [train|t10k]-images-idx3-ubyte.gz: images
- [train|t10k]-labels-idx1-ubyte.gz: digit labels, copied from original MNIST
- [train|t10k]-pert-idx1-ubyte.gz: perturbation labels
    - 0: plain; 1: thinned; 2: thickened; 3: swollen; 4: fractured.
- [train|t10k]-morpho.csv: morphometrics table, with columns:
    - 'index': index of the corresponding digit (for convenience, although rows
      are written in order)
    - 'area' (px²), 'length' (px), 'thickness' (px), 'slant' (rad), 'width'
      (px), 'height' (px): calculated morphometrics

As in the original MNIST, 'train' and 't10k' refer to the 60,000 training and
10,000 test samples, respectively. The data is distributed in the same format
(IDX-encoded uint8 arrays), described in http://yann.lecun.com/exdb/mnist.

More information, code, and the download links for the other Morpho-MNIST
datasets can be found in our GitHub repository:
https://github.com/dccastro/Morpho-MNIST

Please consider citing the accompanying paper if using this data in your
publications:

Castro, Daniel C., Tan, Jeremy, Kainz, Bernhard, Konukoglu, Ender, and Glocker,
  Ben (2018). Morpho-MNIST: Quantitative Assessment and Diagnostics for
  Representation Learning. arXiv preprint arXiv:1809.10780.
  https://arxiv.org/abs/1809.10780

Contact: Daniel Coelho de Castro <dcdecastro@gmail.com>
