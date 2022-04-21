# 20220421
# hexiaohao
# clip 第三方训练代码阅读笔记
# 利用别的预训练模型去 wrap clip

import torch
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from data.text_image_dm import TextImageDataModule
from models import CustomCLIPWrapper
from torchvision.models import resnet50
from transformers import AutoTokenizer, AutoModel


def main(hparams):
    img_encoder = resnet50(pretrained=True)
    img_encoder.fc = torch.nn.Linear(2048, 768)

    tokenizer = AutoTokenizer.from_pretrained("johngiorgi/declutr-sci-base")
    txt_encoder = AutoModel.from_pretrained("johngiorgi/declutr-sci-base")

    if hparams.minibatch_size < 1:
        hparams.minibatch_size = hparams.batch_size

    # 使用定制化 clip_wrapper
    model = CustomCLIPWrapper(img_encoder, txt_encoder, hparams.minibatch_size, avg_word_embs=True)
    dm = TextImageDataModule.from_argparse_args(hparams, custom_tokenizer=tokenizer)
    trainer = Trainer.from_argparse_args(hparams, precision=16, max_epochs=32)
    trainer.fit(model, dm)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--minibatch_size', type=int, default=0)
    parser = TextImageDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
