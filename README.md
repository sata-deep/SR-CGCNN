<!-- ****** Modified by Satadeep Bhattacharjee **** -->

# SR-CGCNN Shared Recurrent Convolution in Crystal Graph Neural Networks for Materials Property Prediction

This repository contains the shared-recurrent Crystal Graph Convolutional
Neural Network code and lightweight dataset downloaders.


<img width="1672" height="941" alt="Arch" src="https://github.com/user-attachments/assets/83412696-7cef-4dff-943f-a8614c2139ae" />




## Repository layout

```text
.
├── README.md
├── SR-CGCNN/
│   ├── main.py
│   ├── predict.py
│   └── cgcnn/
│       ├── __init__.py
│       ├── atom_init.json
│       ├── data.py
│       └── model.py
├── FE/
│   └── download.py
└── Band-Gap/
    └── download.py
```

The original data files, training runs, pycache files, and manuscript
documentation are intentionally not included.

## Dependencies

Install the Python packages needed by training, prediction, and the downloaders:

```bash
pip install numpy scikit-learn torch pymatgen mp-api
```

## Download datasets

Set a Materials Project API key before running either downloader:

```bash
export MP_API_KEY="your-api-key"
python FE/download.py --limit 1000 --stable-only
python Band-Gap/download.py --limit 1000 --stable-only
```

Each downloader writes a CGCNN-style dataset into its directory:
`id_prop.csv`, `atom_init.json`, and `mp-*.cif` files.

## Train

Run training from the repository root. For example:

```bash
python SR-CGCNN/main.py FE \
  --task regression \
  --conv-mode shared_recurrent \
  --n-conv 3 \
  --epochs 300 \
  --optim Adam \
  --lr 0.001 \
  --batch-size 256 \
  --run-dir FE/runs/recurrent_3steps
```

Standard CGCNN-style convolution is also available:

```bash
python SR-CGCNN/main.py Band-Gap \
  --task regression \
  --conv-mode standard \
  --n-conv 3 \
  --epochs 300 \
  --optim Adam \
  --lr 0.001 \
  --batch-size 256 \
  --run-dir Band-Gap/runs/standard_3conv
```

## Predict

Use a trained checkpoint with a directory containing CIF files plus
`id_prop.csv` and `atom_init.json`:

```bash
python SR-CGCNN/predict.py FE/runs/recurrent_3steps/model_best.pth.tar FE
```

## Citation

If you use this code, please cite:

```bibtex
@article{bhattacharjee2026sr,
  title={SR-CGCNN: Shared Recurrent Convolution in Crystal Graph Neural Networks for Materials Property Prediction},
  author={Bhattacharjee, Satadeep},
  journal={arXiv preprint arXiv:2605.01304},
  year={2026}
}
```

## License

This repository contains code derived from the original CGCNN software by
Tian Xie, released under the MIT License. The original copyright notice and
license terms are preserved in [LICENSE](LICENSE).
