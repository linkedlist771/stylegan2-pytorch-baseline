## 1. Porcess Dataset

# 对 gan 来说标签不重要。

```bash
root_path/
└── all/
    ├── image1.jpg
    ├── image2.png
    ├── ...
```

```bash
python preprocess.py --out ./lmdb_deposition_dataset --size 512  --n_worker 8 --resample lanczos ./deposition_dataset

```

```bash
128,256,512,1024
```

## 2. Train....

```bash

```
