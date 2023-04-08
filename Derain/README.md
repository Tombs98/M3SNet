
## Training
- Download the [Datasets](Datasets/README.md)

- Train the model with default arguments by running

```
python train.py
```


## Evaluation

1. Download the [model](https://drive.google.com/drive/folders/1y4BEX7LagtXVO98ZItSbJJl7WWM3gnbD?usp=share_link) and place it in `./pretrained_models/`

2. Download test datasets (Test100, Rain100H, Rain100L, Test1200) from [here](https://drive.google.com/drive/folders/1PDWggNh8ylevFmrjo-JEvlmqsDlWWvZs?usp=sharing) and place them in `./Datasets/Synthetic_Rain_Datasets/test/`

3. Run
4. 
```
python test.py
```

#### To reproduce PSNR/SSIM scores of the paper, run

```
python evaluate_PSNR_SSIM.py
```
