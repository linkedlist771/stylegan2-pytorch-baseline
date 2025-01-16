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
python prepare_data.py --out ./lmdb_deposition_dataset --size 512  --n_worker 8 --resample lanczos  /mnt/c/Users/23174/Desktop/GitHub\ Project/data-efficient-gans-baseline/data/deposition_data_processed_stylegan/ 

```

```bash
128,256,512,1024
```

## 2. Train....

```bash

```
