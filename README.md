# [GVA: Guided Visual Attention Approach for Automatic Image Caption Generation](https://link.springer.com/article/10.1007/s00530-023-01249-w)
## Requirements

- Python 2.7
- Java 1.8.0
- PyTorch 0.4.0
- cider (already been added as a submodule)
- coco-caption (already been added as a submodule)
- tensorboardX
- ### Prepare data

See details in `data/README.md`.

You should also preprocess the dataset and get the cache for calculating cider score for [SCST](https://arxiv.org/abs/1612.00563):

```bash
$ python scripts/prepro_ngrams.py --input_json data/dataset_coco.json --dict_json data/cocotalk_attr.json --output_pkl data/coco-train-new --split train
```
### Start training

```bash
$ base train.sh
```
We train our model on 2 TitanXp GPUs, you can change the batch_size and gpu_nums in xe_train.sh  to train the model on your own hardware.
See `opts.py` for the options. The pretrained models can be downloaded [here](https://drive.google.com/open?id=1-JRl_3Vf0tzyOgEwfCH6yNDIdCyobNvH).


### Evaluation
You should enter the model id and checkpoint number in eval.py before evaluation. Note that the beam size can only be altered in AttModel_GVA.py and CaptionModel.py manually. This is because opt is not compatible with multi-GPU training.
```bash
$ CUDA_VISIBLE_DEVICES=0 python eval.py  --num_images -1 --language_eval 1 --batch_size 100 --split test
```



## Reference

If you find this repo helpful, please consider citing:
@article{hossen2024gva,
  title={GVA: guided visual attention approach for automatic image caption generation},
  author={Hossen, Md Bipul and Ye, Zhongfu and Abdussalam, Amr and Hossain, Md Imran},
  journal={Multimedia Systems},
  volume={30},
  number={1},
  pages={50},
  year={2024},
  publisher={Springer}
}

Hossen, M.B., Ye, Z., Abdussalam, A. et al. GVA: guided visual attention approach for automatic image caption generation. Multimedia Systems 30, 50 (2024). https://doi.org/10.1007/s00530-023-01249-w

## Acknowledgements

This repository is based on [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch), and you may refer to it for more details about the code.
