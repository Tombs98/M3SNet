## Training
- Download the [Datasets](Datasets/README.md)

- Train the model with default arguments by running

```
python train.py
```

## Evaluation

### Download the [model](https://drive.google.com/file/d/1JJUwbH5cYEaTvtQ8jGF406ZKfDqXyNFD/view?usp=share_link) and place it in ./pre-trained/

#### Testing on GoPro dataset
- Download [images](https://drive.google.com/drive/folders/1a2qKfXWpNuTGOm2-Jex8kfNSzYJLbqkf?usp=sharing) of GoPro and place them in `./Datasets/GoPro/test/`
- Run
```
python test.py --dataset GoPro
```

#### Testing on HIDE dataset
- Download [images](https://drive.google.com/drive/folders/1nRsTXj4iTUkTvBhTcGg8cySK8nd3vlhK?usp=sharing) of HIDE and place them in `./Datasets/HIDE/test/`
- Run
```
python test.py --dataset HIDE
```
`
```

#### To reproduce PSNR scores of the paper on GoPro and HIDE datasets, run 
```
python test_res.py 
```

#### To reproduce SSIM scores of the paper on GoPro and HIDE datasets, run 
```
python eval.py
```
