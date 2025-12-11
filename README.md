## Requirements
- Python 3.8+
- PyTorch 2.0+
- torchvision
- Additional dependencies:
```bash
pip install ftfy tqdm scipy pandas numpy Pillow tensorboard openai-clip
```

## Datasets
Download the following datasets and place them in the `./data` directory:
- [**AGIQA-1K**](https://github.com/lcysyzxdxc/AGIQA-1k-Database)
- [**AGIQA-3K**](https://github.com/lcysyzxdxc/AGIQA-3k-Database)
- [**AIGCIQA2023**](https://github.com/wangjiarui153/AIGCIQA2023)

### Training
```bash
bash start_training.sh
```

### Testing
```bash
bash start_testing.sh
```
