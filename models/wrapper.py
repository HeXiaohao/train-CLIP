# 20220421
# hexiaohao
# clip 第三方训练代码阅读笔记
# 定义整个 clip 的基础架构 wrapper。同时定义了视觉和文本的 backbone

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import math
import yaml
import copy
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from .model import CLIP


class CLIPWrapper(pl.LightningModule):
    def __init__(self,
                 model_name: str,
                 config: dict,
                 minibatch_size: int
                 ):
        """A lightning wrapper for a CLIP model as specified in the paper.

        Args:
            model_name (str): A case sensitive visual model name.
            config (dict): A dictionary containing the CLIP instantiation parameters.
        """
        super().__init__()

        self.model_name = model_name
        self.model = CLIP(**config)
        self.minibatch_size = minibatch_size
        self.isViT = 'ViT' in self.model_name

        self.automatic_optimization = False
    
    # 这个函数是用来计算总共需要训练多少个 steps
    # Sourced from https://github.com/PyTorchLightning/pytorch-lightning/issues/5449
    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = self.train_dataloader()
        if self.trainer.max_steps:
            return self.trainer.max_steps

        dataset_size = len(dataset)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_batch_size = dataset.batch_size * self.trainer.accumulate_grad_batches * num_devices
        return (dataset_size // effective_batch_size) * self.trainer.max_epochs

    # 感觉是每一个 batch 的具体训练？
    # Training loss: https://github.com/openai/CLIP/issues/83
    # Mini-batching thanks to https://github.com/crowsonkb / https://twitter.com/RiversHaveWings
    # Multi-GPU support: https://github.com/MicPie/clasp
    def training_step(self, train_batch, idx):
        # 获取训练的优化器
        # get optimizers and scheduler
        optimizer = self.optimizers()

        # 获取一个 batch 的训练数据
        image, text = train_batch

        # 根据 minibatch_size，把 1 个 batch_size 的数据分成若干块
        n = math.ceil(len(image) // self.minibatch_size)
        image_mbs = torch.chunk(image, n)
        text_mbs = torch.chunk(text, n)

        # 根据视觉和文本 backbone，分别计算两种模态的 features。并且计算相关的 acc，由于不需要训练，因此不需要开梯度反向传播
        # calculate original statistics
        with torch.no_grad():
            ims = [F.normalize(self.model.encode_image(im), dim=1) for im in image_mbs]
            txt = [F.normalize(self.model.encode_text(t), dim=1) for t in text_mbs]

            # 没有很看明白这个用来干什么？
            # gather from all GPUs
            ims = self.all_gather(torch.cat(ims))
            txt = self.all_gather(torch.cat(txt))

            # list 会把 tensor 的第 0 维 list 化
            if len(ims.shape) == 3:
                ims = list(ims)
                txt = list(txt)
            else:
                ims = [ims]
                txt = [txt]

            # @ 是矩阵乘法。* 是矩阵点乘
            image_logits = torch.cat(ims) @ torch.cat(txt).t() * self.model.logit_scale.exp()
            ground_truth = torch.arange(len(image_logits)).long().to(image_logits.device)
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth)).div(2)
            # 这部分是用来计算 image 侧和 text 侧的 acc 的
            acc_i = (torch.argmax(image_logits, 1) == ground_truth).sum()
            acc_t = (torch.argmax(image_logits, 0) == ground_truth).sum()
            self.log_dict({'loss': loss / len(ims), 'acc': (acc_i + acc_t) / 2 / len(image) / len(ims)}, prog_bar=True)

        # 如果指定多个 optimizer，就只使用第一个
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        optimizer.zero_grad()

        # 计算图像侧的 loss
        # image loss
        for j, mb in enumerate(image_mbs):
            images_tmp = copy.deepcopy(ims)
            images_tmp[self.global_rank][j*self.minibatch_size:(j+1)*self.minibatch_size] = F.normalize(self.model.encode_image(mb), dim=1)
            image_logits = torch.cat(images_tmp) @ torch.cat(txt).t() * self.model.logit_scale.exp()
            ground_truth = torch.arange(len(image_logits)).long().to(image_logits.device)
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth))/2
            self.manual_backward(loss)

        # 计算文本侧的 loss
        # text loss
        for j, mb in enumerate(text_mbs):
            text_tmp = copy.deepcopy(txt)
            text_tmp[self.global_rank][j*self.minibatch_size:(j+1)*self.minibatch_size] = F.normalize(self.model.encode_text(mb), dim=1)
            image_logits = torch.cat(ims) @ torch.cat(text_tmp).t() * self.model.logit_scale.exp()
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth))/2
            self.manual_backward(loss)

        optimizer.step()
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()
        self.model.logit_scale.data.clamp_(-np.log(100), np.log(100))

    # 一次 val 测试。计算 val 的 loss
    def validation_step(self, val_batch, idx):
        image, text = val_batch
        image_logits, text_logits = self.forward(image, text)
        ground_truth = torch.arange(len(image_logits))
        loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(text_logits, ground_truth)).div(2)
        self.log('val_loss', loss)

    # 构建 optimizers 和 lr_scheduler
    def configure_optimizers(self):
        lr = {
            "RN50": 5e-4,
            "RN101": 5e-4,
            "RN50x4": 5e-4,
            "RN50x16": 4e-4,
            "RN50x64": 3.6e-4,
            "ViT-B/32": 5e-4,
            "ViT-B/16": 5e-4,
            "ViT-L/14": 4e-4,
            "ViT-L/14-336px": 2e-5
        }[self.model_name]

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(
                0.9,
                0.98 if self.isViT else 0.999
            ),
            eps=1e-6 if self.isViT else 1e-8,
            weight_decay=0.2
        )

        # Source: https://github.com/openai/CLIP/issues/107
        # Use pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.num_training_steps,
            cycle_mult=1.0,
            max_lr=lr,
            min_lr=0,
            warmup_steps=2000
        )

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


class CustomCLIPWrapper(CLIPWrapper):
    def __init__(self,
                 image_encoder,
                 text_encoder,
                 minibatch_size,
                 learning_rate=3e-3,
                 kl_coeff=1.0,
                 avg_word_embs=False
                 ):
        with open('models/configs/RN.yaml') as fin:
            config = yaml.safe_load(fin)['RN50']
        super().__init__('RN50', config, minibatch_size)
        del self.model.visual
        del self.model.transformer
        self.model.visual = image_encoder
        self.model.transformer = text_encoder
        self.learning_rate = learning_rate
        self.avg_word_embs = avg_word_embs
        self.sink_temp = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # init self-distillation model
        self.teacher = copy.deepcopy(self.model)
        self.kl_coeff = kl_coeff

    # 一个 batch 的训练
    def training_step(self, train_batch, idx):
        # get optimizers and scheduler
        optimizer = self.optimizers()

        image, text = train_batch
        n = math.ceil(len(image) // self.minibatch_size)
        image_mbs = torch.chunk(image, n)

        # 就是把一个 tensor 按照顺序拆开
        # >>> torch.chunk(torch.arange(4), 2)
        # (tensor([0, 1]), tensor([2, 3]))
        text_mbs_ids = torch.chunk(torch.arange(len(image)), n)

        # adjust embedding dictionaries
        # 每个 tokenizer 输出的东西有好几个，比如 encode_plus 会输出一个字典，分别为 'input_ids', 'token_type_ids', 'attention_mask' 对应的编码
        text_mbs = []
        for s in text_mbs_ids:
            d = {}
            for key in list(text.keys()):
                d[key] = text[key][s]
            text_mbs.append(d)

        # 计算训练之前的统计特征，包括 loss 和 acc
        # calculate original statistics
        with torch.no_grad():
            # self.model 调用的是包装的 clip
            ims = [F.normalize(self.model.encode_image(im), dim=1) for im in image_mbs]

            # self.encode_text 调用的是当前的 encoder，而且这里没用 teacher
            txt = [F.normalize(self.encode_text(t), dim=1) for t in text_mbs]
            # gather from all GPUs
            ims = self.all_gather(torch.cat(ims))
            txt = self.all_gather(torch.cat(txt))

            if len(ims.shape) == 3:
                ims = list(ims)
                txt = list(txt)
            else:
                ims = [ims]
                txt = [txt]

            # 在不进行梯度反向传播的前提下，计算 loss
            # 这部分代码和前面的 CLIPWrapper 是一样的
            image_logits_notemp = torch.cat(ims) @ torch.cat(txt).t()
            image_logits = image_logits_notemp * self.model.logit_scale.exp()
            ground_truth = torch.arange(len(image_logits)).long().to(image_logits.device)
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth)).div(2)
            acc_i = (torch.argmax(image_logits, 1) == ground_truth).sum()
            acc_t = (torch.argmax(image_logits, 0) == ground_truth).sum()

            # 计算 teacher 网络的相关 encode emb
            # calculate teacher
            teacher_ims = [F.normalize(self.teacher.encode_image(im), dim=1) for im in image_mbs]
            teacher_txt = [F.normalize(self.encode_text(t, teacher=True), dim=1) for t in text_mbs]

            teacher_ims = self.all_gather(torch.cat(teacher_ims))
            teacher_txt = self.all_gather(torch.cat(teacher_txt))

            if len(teacher_ims.shape) == 3:
                teacher_ims = list(teacher_ims)
                teacher_txt = list(teacher_txt)
            else:
                teacher_ims = [teacher_ims]
                teacher_txt = [teacher_txt]

            # 计算 teacher 网络在 Image 侧和 Text 侧的矩阵相似性
            sim_ii, sim_tt, sim_it, sim_ti = self.compute_similarities(torch.cat(teacher_ims), torch.cat(teacher_txt))

            # 计算当前网络的 image-text emb 相似性矩阵，和 teacher 网络的相似性矩阵的差距
            # optimal transport
            img_cost = - (sim_ii + sim_tt + sim_it)
            txt_cost = - (sim_ii + sim_tt + sim_ti)
            img_target = self.sinkhorn(img_cost)
            txt_target = self.sinkhorn(txt_cost)
            loss += (F.kl_div(F.log_softmax(image_logits_notemp * self.sink_temp, dim=-1), img_target, reduction='batchmean') + F.kl_div(F.log_softmax(image_logits_notemp.t() * self.sink_temp, dim=-1), txt_target, reduction='batchmean')) / 2 * self.kl_coeff
            self.log_dict({'loss': loss / len(ims), 'acc': (acc_i + acc_t) / 2 / len(image) / len(ims)}, prog_bar=True)

        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        optimizer.zero_grad()

        # image loss
        for j, mb in enumerate(image_mbs):
            images_tmp = copy.deepcopy(ims)
            images_tmp[self.global_rank][j*self.minibatch_size:(j+1)*self.minibatch_size] = F.normalize(self.model.encode_image(mb), dim=1)
            image_logits_notemp = torch.cat(images_tmp) @ torch.cat(txt).t()
            image_logits = image_logits_notemp * self.model.logit_scale.exp()
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth))/2
            loss += (F.kl_div(F.log_softmax(image_logits_notemp * self.sink_temp, dim=-1), img_target, reduction='batchmean') + F.kl_div(F.log_softmax(image_logits_notemp.t() * self.sink_temp, dim=-1), txt_target, reduction='batchmean')) / 2 * self.kl_coeff
            self.manual_backward(loss)

        # text loss
        for j, mb in enumerate(text_mbs):
            text_tmp = copy.deepcopy(txt)
            text_tmp[self.global_rank][j*self.minibatch_size:(j+1)*self.minibatch_size] = F.normalize(self.encode_text(mb), dim=1)
            image_logits_notemp = torch.cat(ims) @ torch.cat(text_tmp).t()
            image_logits = image_logits_notemp * self.model.logit_scale.exp()
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth))/2
            loss += (F.kl_div(F.log_softmax(image_logits_notemp * self.sink_temp, dim=-1), img_target, reduction='batchmean') + F.kl_div(F.log_softmax(image_logits_notemp.t() * self.sink_temp, dim=-1), txt_target, reduction='batchmean')) / 2 * self.kl_coeff
            self.manual_backward(loss)

        optimizer.step()
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()
        self.model.logit_scale.data.clamp_(-np.log(100), np.log(100))

        # 这两步 teacher 网络相关的步骤是新增的
        self.sink_temp.data.clamp_(-np.log(100), np.log(100))
        self.update_teacher()

    # CustomCLIPWrapper 重新定制的 encode_text 方法
    def encode_text(self, inputs, teacher=False):
        # bert 的输出长度和输入的词汇长度基本是一一对应的，一般只取 cls 头的输出
        # 这里可以选择把 bert 产生的 word 的 emb 都做一个平均
        if self.avg_word_embs:
            sequence_output = self.teacher.transformer(**inputs)[0] if teacher else self.model.transformer(**inputs)[0]

            embeddings = torch.sum(
                sequence_output * inputs["attention_mask"].unsqueeze(-1), dim=1
            ) / torch.clamp(torch.sum(inputs["attention_mask"], dim=1, keepdims=True), min=1e-9)

            return embeddings
        else:
            return self.teacher.transformer(**inputs)[1] if teacher else self.model.transformer(**inputs)[1]

    # 计算 teacher 网络的 Image 侧和 Text 侧的矩阵相似性
    def compute_similarities(self, I_emb, T_emb):
        sim_ii, sim_tt = I_emb @ I_emb.t(), T_emb @ T_emb.t()
        sim_it, sim_ti = I_emb @ T_emb.t(), T_emb @ I_emb.t()
        return sim_ii, sim_tt, sim_it, sim_ti

    # 缓慢更新 teacher 网络的参数
    def update_teacher(self):
        for teacher, student in zip(self.teacher.parameters(), self.model.parameters()):
            teacher.data.copy_(self.ema(student.data, teacher.data))

    def ema(self, s, t):
        return s * (1 - 0.999) + t * 0.999

    def forward(self, images, text):
        logits = F.normalize(self.model.encode_image(images), dim=1) @ F.normalize(self.encode_text(text), dim=1).t() * self.model.logit_scale.exp()
        return logits, logits.t()

    # 暂时看不懂。。。无监督学习相关的一个函数
    # Sourced from: https://github.com/facebookresearch/swav/blob/5e073db0cc69dea22aa75e92bfdd75011e888f28/main_swav.py#L354
    def sinkhorn(self, out):
        Q = torch.exp(out / 0.05).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1]  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(3):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()

    def configure_optimizers(self):
        lr = self.learning_rate

        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=lr,
            momentum=0.9
        )

        # Source: https://github.com/openai/CLIP/issues/107
        # Use pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.num_training_steps,
            cycle_mult=1.0,
            max_lr=lr,
            min_lr=0,
            warmup_steps=2000
        )

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
