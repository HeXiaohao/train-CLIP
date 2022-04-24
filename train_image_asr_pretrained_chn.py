# 20220424
# hexiaohao
# 图文匹配+图像预训练+中文预训练

import torch
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from data.text_image_tfrecord_dm import TextImageDataModule
from models import CustomCLIPWrapper
from torchvision.models import resnet50
from transformers import AutoTokenizer, AutoModel, BertModel, BertConfig, BertTokenizer


def main(hparams):
    img_encoder = resnet50(pretrained=False)
    img_encoder.load_state_dict(torch.load(hparams.img_weights_path))
    img_encoder.fc = torch.nn.Linear(2048, 768)

    tokenizer = BertTokenizer.from_pretrained(hparams.txt_weights_path)
    txt_encoder = BertModel.from_pretrained(hparams.txt_weights_path)

    if hparams.minibatch_size < 1:
        hparams.minibatch_size = hparams.batch_size

    # 使用定制化 clip_wrapper
    model = CustomCLIPWrapper(img_encoder, txt_encoder, hparams.minibatch_size, avg_word_embs=True)
    dm = TextImageDataModule.from_argparse_args(hparams, custom_tokenizer=tokenizer)
    # gpus=4 四卡训练
    trainer = Trainer.from_argparse_args(hparams, precision=16, max_epochs=32, gpus=4)
    trainer.fit(model, dm)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--minibatch_size', type=int, default=0)
    parser.add_argument('--img_weights_path', type=str, required=True, help='path of model weights for img_encoder')
    parser.add_argument('--txt_weights_path', type=str, required=True, help='path of model weights for txt_encoder')
    parser = TextImageDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
