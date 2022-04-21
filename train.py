# 20220421
# hexiaohao
# clip 第三方训练代码阅读笔记
# 从头开始训练 clip。完全对照 clip 论文中的参数设置

import yaml
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from data.text_image_dm import TextImageDataModule
from models import CLIPWrapper


def main(hparams):
    # 选择视觉 backbone
    config_dir = 'models/configs/ViT.yaml' if 'ViT' in hparams.model_name else 'models/configs/RN.yaml'

    # 加载视觉 backbone 参数
    with open(config_dir) as fin:
        config = yaml.safe_load(fin)[hparams.model_name]

    # 这个参数有什么用？
    if hparams.minibatch_size < 1:
        hparams.minibatch_size = hparams.batch_size

    # 定义整个 clip 的基础架构 wrapper。同时定义了视觉和文本的 backbone
    model = CLIPWrapper(hparams.model_name, config, hparams.minibatch_size)
    del hparams.model_name
    dm = TextImageDataModule.from_argparse_args(hparams)
    trainer = Trainer.from_argparse_args(hparams, precision=16, max_epochs=32)
    trainer.fit(model, dm)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--minibatch_size', type=int, default=0)

    # TextImageDataModule 的 parse 参数
    parser = TextImageDataModule.add_argparse_args(parser)

    # Trainer 的 parse 参数
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
