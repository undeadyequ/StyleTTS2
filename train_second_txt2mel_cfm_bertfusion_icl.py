"""Second-stage training for CFMDecoderV4 (in-context learning via masked-mel conditioning).

Key differences from train_second_txt2mel_cfm_bertfusion.py:
  - Uses CFMDecoderV4 (flow_matching_v4.py), selected by `use_v4: true` in cfm_config.
  - style_encoder is NOT used for decoder conditioning (c removed from compute_loss/forward).
    style_encoder is kept for diffusion training (gs → s_trg).
  - predictor_encoder and sampler are kept (predict prosody / diffusion style).
  - Training compute_loss: model.decoder.compute_loss(gt, mu=en, seq_style=pe)
    (no mask, c, p_mask).
  - Validation inference: ICL mode — utterance split ref=30% (2nd portion) / tgt=70% (1st portion);
    matches F5-TTS mask_ratio_min=0.7. model.decoder called with mu_tgt, cond_ref, mu_ref,
    seq_style_tgt, seq_style_ref.
"""

# load packages
import os.path
import random
import time
import click
import shutil
import traceback
import warnings

import torch

warnings.simplefilter('ignore')
from torch.utils.tensorboard import SummaryWriter

from meldataset2 import build_dataloader

from Utils.PLBERT.util import load_plbert

from models_txt2mel_cfm import *
from losses import *
from utils import *
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from optimizers import build_optimizer
from attrdict import AttrDict
from Modules.hifi_gan.vocoder import Generator
import glob
from utils import r1_reg, adv_loss, get_cut_phonemes_by_cut_f2p_attn
import json

# simple fix for dataparallel that allows access to class attributes
class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

import logging
from logging import StreamHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

@click.command()
@click.option('-p', '--config_path', default='Configs/config_libritts_txt2mel_cfm_v37.yml', type=str)
def main(config_path):
    torch.manual_seed(0)
    config = yaml.safe_load(open(config_path))

    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    writer = SummaryWriter(log_dir + "/tensorboard")

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)

    batch_size = config.get('batch_size', 10)

    epochs = config.get('epochs_2nd', 200)
    save_freq = config.get('save_freq', 2)
    data_ratio = config.get('data_ratio', 1)
    log_interval = config.get('log_interval', 10)
    saving_epoch = config.get('save_freq', 2)

    data_params = config.get('data_params', None)
    sr = config['preprocess_params'].get('sr', 24000)
    train_path = data_params['train_data']
    val_path = data_params['val_data']
    root_path = data_params['root_path']
    min_length = data_params['min_length']
    OOD_data = data_params['OOD_data']

    max_len = config.get('max_len', 200)

    loss_params = Munch(config['loss_params'])
    diff_epoch = loss_params.diff_epoch
    joint_epoch = loss_params.joint_epoch

    optimizer_params = Munch(config['optimizer_params'])

    # load data
    train_list, val_list = get_data_path_list(train_path, val_path)
    device = 'cuda'
    if data_ratio < 1:
        train_list = train_list[:int(len(train_list) * data_ratio)]

    train_dataloader = build_dataloader(train_list,
                                        root_path,
                                        OOD_data=OOD_data,
                                        min_length=min_length,
                                        batch_size=batch_size,
                                        num_workers=2,
                                        dataset_config={},
                                        device=device)

    val_dataloader = build_dataloader(val_list,
                                      root_path,
                                      OOD_data=OOD_data,
                                      min_length=min_length,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=0,
                                      device=device,
                                      dataset_config={})

    # load pretrained ASR model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    # load pretrained F0 model
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)

    # load PL-BERT model
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)

    # build model
    model_params = recursive_munch(config['model_params'])
    multispeaker = model_params.multispeaker
    cond_prosody_type = model_params.get("cond_prosody_type", "predict")
    cfm_params = recursive_munch(config['cfm_config'])
    cfg_strength_val = 3 if cfm_params.get('cfg_dropout', 0) > 0 else None
    pitch_min, pitch_max, energy_min, energy_max = tuple(model_params.pe_min_max)

    model = build_model_v4(model_params, config['cfm_config'], text_aligner, pitch_extractor, plbert)
    style_dim = model_params.style_dim

    _ = [model[key].to(device) for key in model]

    start_epoch = 0
    load_pretrained = config.get('pretrained_model', '') != '' and config.get('second_stage_load_pretrained', False)

    # load first stage
    if not load_pretrained:
        if config.get('first_stage_path', '') != '':
            first_stage_path = osp.join(log_dir, config.get('first_stage_path', 'first_stage.pth'))
            print('Loading the first stage model at %s ...' % first_stage_path)
            model, _, start_epoch, iters = load_checkpoint(model,
                                                           None,
                                                           first_stage_path,
                                                           load_only_params=True,
                                                           ignore_modules=['bert', 'bert_encoder', 'predictor',
                                                                           'predictor_encoder',
                                                                           'diffusion'])
            diff_epoch += start_epoch
            joint_epoch += start_epoch
            epochs += start_epoch
        else:
            raise ValueError('You need to specify the path to the first stage model.')

    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False
    )
    scheduler_params = {
        "max_lr": optimizer_params.lr,
        "pct_start": float(0),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }
    scheduler_params_dict = {key: scheduler_params.copy() for key in model}
    scheduler_params_dict['bert']['max_lr'] = optimizer_params.bert_lr * 2
    scheduler_params_dict['decoder']['max_lr'] = optimizer_params.ft_lr * 2
    scheduler_params_dict['style_encoder']['max_lr'] = optimizer_params.ft_lr * 2
    optimizer = build_optimizer({key: model[key].parameters() for key in model},
                                scheduler_params_dict=scheduler_params_dict, lr=optimizer_params.lr)

    # adjust BERT learning rate
    for g in optimizer.optimizers['bert'].param_groups:
        g['betas'] = (0.9, 0.99)
        g['lr'] = optimizer_params.bert_lr
        g['initial_lr'] = optimizer_params.bert_lr
        g['min_lr'] = 0
        g['weight_decay'] = 0.01

    # adjust acoustic module learning rate
    for module in ["decoder", "style_encoder"]:
        for g in optimizer.optimizers[module].param_groups:
            g['betas'] = (0.0, 0.99)
            g['lr'] = optimizer_params.ft_lr
            g['initial_lr'] = optimizer_params.ft_lr
            g['min_lr'] = 0
            g['weight_decay'] = 1e-4

    if load_pretrained:
        model, optimizer, start_epoch, iters = load_checkpoint(model, optimizer, config['pretrained_model'],
                                                               load_only_params=config.get('load_only_params', True))

    n_down = model.text_aligner.n_down

    best_loss = float('inf')
    iters = 0
    torch.cuda.empty_cache()
    print('BERT', optimizer.optimizers['bert'])
    print('decoder', optimizer.optimizers['decoder'])

    running_std = []
    for epoch in range(start_epoch, epochs):
        running_loss = 0
        start_time = time.time()

        _ = [model[key].eval() for key in model]
        model.predictor.train()
        model.bert_encoder.train()
        model.bert.train()

        for i, batch in enumerate(train_dataloader):
            waves = batch[0]
            batch = [b.to(device) for b in batch[1:]]
            texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels, uv_masks = batch

            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                mel_mask = length_to_mask(mel_input_length).to(device)
                text_mask = length_to_mask(input_lengths).to(texts.device)
                try:
                    _, _, s2s_attn = model.text_aligner(mels, mask, texts)
                    s2s_attn = s2s_attn.transpose(-1, -2)
                    s2s_attn = s2s_attn[..., 1:]
                    s2s_attn = s2s_attn.transpose(-1, -2)
                except:
                    continue
                mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                t_en = model.text_encoder(texts, input_lengths, text_mask)
                asr = (t_en @ s2s_attn_mono)
                d_gt = s2s_attn_mono.sum(axis=-1).detach()

                # Get reference prosodic style for diffusion (multispeaker)
                # ref_ss (acoustic style) is not used — speaker identity is supplied to
                # the ICL decoder via cond_ref, so only ref_sp (prosodic) is needed.
                if multispeaker and epoch >= diff_epoch:
                    ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))

            # Prosodic style for diffusion target (style_dim only; no acoustic style_encoder)
            ss = []
            for bib in range(len(mel_input_length)):
                mel = mels[bib, :, :mel_input_length[bib]]
                s = model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))
                ss.append(s)
            s_dur = torch.stack(ss).squeeze()
            s_trg = s_dur.detach()

            bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

            if epoch >= diff_epoch:
                num_steps = np.random.randint(3, 5)
                if model_params.diffusion.dist.estimate_sigma_data:
                    model.diffusion.diffusion.sigma_data = s_trg.std(axis=-1).mean().item()
                    running_std.append(model.diffusion.diffusion.sigma_data)
                if multispeaker:
                    s_preds = sampler(noise=torch.randn_like(s_trg).unsqueeze(1).to(device),
                                      embedding=bert_dur,
                                      embedding_scale=1,
                                      features=ref_sp,
                                      embedding_mask_proba=0.1,
                                      num_steps=num_steps).squeeze(1)
                    loss_diff = model.diffusion(s_trg.unsqueeze(1), embedding=bert_dur, features=ref_sp).mean()
                    loss_sty = F.l1_loss(s_preds, s_trg.detach())
                else:
                    s_preds = sampler(noise=torch.randn_like(s_trg).unsqueeze(1).to(device),
                                      embedding=bert_dur,
                                      embedding_scale=1,
                                      embedding_mask_proba=0.1,
                                      num_steps=num_steps).squeeze(1)
                    loss_diff = model.diffusion.diffusion(s_trg.unsqueeze(1), embedding=bert_dur).mean()
                    loss_sty = F.l1_loss(s_preds, s_trg.detach())
            else:
                loss_sty = 0
                loss_diff = 0

            d, p = model.predictor(d_en, s_dur, input_lengths, s2s_attn_mono, text_mask)
            mel_len = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)
            mel_len_st = int(mel_input_length.min().item() / 2 - 1)
            en, gt, st, p_en, wav, s2s = [], [], [], [], [], []

            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)
                random_start = np.random.randint(0, mel_length - mel_len)
                en.append(asr[bib, :, random_start:random_start + mel_len])
                s2s.append(s2s_attn_mono[bib, :, random_start:random_start + mel_len])
                p_en.append(p[bib, :, random_start:random_start + mel_len])
                gt.append(mels[bib, :, (random_start * 2):((random_start + mel_len) * 2)])
                y = waves[bib][(random_start * 2) * 300:((random_start + mel_len) * 2) * 300]
                wav.append(torch.from_numpy(y).to(device))

                random_start = np.random.randint(0, mel_length - mel_len_st)
                st.append(mels[bib, :, (random_start * 2):((random_start + mel_len_st) * 2)])

            with torch.no_grad():
                mel_length_cut = torch.where(mel_input_length < mel_len * 2, mel_input_length, torch.tensor(mel_len * 2))
                mel_cut_mask = length_to_mask(mel_length_cut).to('cuda')

            en, p_en, gt, st, s2s = torch.stack(en), torch.stack(p_en), torch.stack(gt).detach(), torch.stack(st).detach(), torch.stack(s2s).detach()
            if gt.size(-1) < 80:
                continue

            # Prosodic style for predictor (no style_encoder for decoder c)
            s_dur = model.predictor_encoder(st.unsqueeze(1) if multispeaker else gt.unsqueeze(1))

            with torch.no_grad():
                F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1))
                N_real = log_norm(gt.unsqueeze(1)).squeeze(1)
            if "bertfusion" in cond_prosody_type:
                F0_trend_drop = True if random.random() < 0.2 else False
                cut_phn_start, cut_phn_end, _, _ = get_phone_range_by_cut_f2p_attn(s2s)
                s2s_cutphone = s2s[:, cut_phn_start:cut_phn_end, :]
                uv_masks_cut = uv_masks[:, cut_phn_start:cut_phn_end]
                trd = model.predictor.trd_encoding(F0_real, s2s_cutphone, uv_masks_cut, drop_trend=F0_trend_drop)
                p_en = torch.cat([p_en, trd], dim=1)

            F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s_dur)
            loss_F0_rec = (F.smooth_l1_loss(F0_real, F0_fake)) / 10
            loss_norm_rec = F.smooth_l1_loss(N_real, N_fake)
            pitch_min, pitch_max = min(pitch_min, torch.min(F0_real).item()), max(pitch_max, torch.max(F0_real).item())
            energy_min, energy_max = min(energy_min, torch.min(N_real).item()), max(energy_max, torch.max(N_real).item())

            pe = torch.cat([N_fake.unsqueeze(1), F0_fake.unsqueeze(1)], dim=1)

            # CFMDecoderV4: no global c, no mask/p_mask args
            optimizer.zero_grad()
            loss_cfm, _ = model.decoder.compute_loss(gt, mu=en, seq_style=pe)

            loss_ce, loss_dur = 0, 0
            for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                _s2s_pred = _s2s_pred[:_text_length, :]
                _text_input = _text_input[:_text_length].long()
                _s2s_trg = torch.zeros_like(_s2s_pred)
                for p_idx in range(_s2s_trg.shape[0]):
                    _s2s_trg[p_idx, :_text_input[p_idx]] = 1
                _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)
                loss_dur += F.l1_loss(_dur_pred[1:_text_length - 1], _text_input[1:_text_length - 1])
                loss_ce += F.binary_cross_entropy_with_logits(_s2s_pred.flatten(), _s2s_trg.flatten())
            loss_ce /= texts.size(0)
            loss_dur /= texts.size(0)
            g_loss = loss_params.lambda_mel * loss_cfm + \
                     loss_params.lambda_F0 * loss_F0_rec + \
                     loss_params.lambda_ce * loss_ce + \
                     loss_params.lambda_norm * loss_norm_rec + \
                     loss_params.lambda_dur * loss_dur + \
                     loss_params.lambda_sty * loss_sty + \
                     loss_params.lambda_diff * loss_diff
            running_loss += loss_cfm
            g_loss.backward()

            if torch.isnan(g_loss):
                pass
            optimizer.step('bert_encoder')
            optimizer.step('bert')
            optimizer.step('predictor')
            optimizer.step('predictor_encoder')

            if epoch >= diff_epoch:
                optimizer.step('diffusion')
            if epoch >= joint_epoch:
                optimizer.step('decoder')
            iters = iters + 1

            if (i + 1) % log_interval == 0:
                logger.info(
                    'Epoch [%d/%d], Step [%d/%d], Loss: %.5f, cfm Loss: %.5f, Dur Loss: %.5f, CE Loss: %.5f, Norm Loss: %.5f, F0 Loss: %.5f, Sty Loss: %.5f, Diff Loss: %.5f '
                    % (epoch + 1, epochs, i + 1, len(train_list) // batch_size, running_loss / log_interval, loss_cfm,
                       loss_dur, loss_ce, loss_norm_rec, loss_F0_rec, loss_sty, loss_diff))

                writer.add_scalar('train/cfm_loss', running_loss / log_interval, iters)
                writer.add_scalar('train/ce_loss', loss_ce, iters)
                writer.add_scalar('train/dur_loss', loss_dur, iters)
                writer.add_scalar('train/norm_loss', loss_norm_rec, iters)
                writer.add_scalar('train/F0_loss', loss_F0_rec, iters)
                writer.add_scalar('train/sty_loss', loss_sty, iters)
                writer.add_scalar('train/diff_loss', loss_diff, iters)
                running_loss = 0
                print('Time elasped:', time.time() - start_time)

        loss_test = 0
        loss_align = 0
        loss_f = 0
        _ = [model[key].eval() for key in model]

        if epoch == start_epoch:
            with open(osp.join(log_dir, "pe_stats.txt"), 'w') as outfile:
                outfile.write(",".join(map(str, [pitch_min, pitch_max, energy_min, energy_max])))

        with torch.no_grad():
            iters_test = 0
            for batch_idx, batch in enumerate(val_dataloader):
                optimizer.zero_grad()
                try:
                    waves = batch[0]
                    batch = [b.to(device) for b in batch[1:]]
                    texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels, uv_masks = batch
                    with torch.no_grad():
                        mask = length_to_mask(mel_input_length // (2 ** n_down)).to('cuda')
                        text_mask = length_to_mask(input_lengths).to(texts.device)
                        _, _, s2s_attn = model.text_aligner(mels, mask, texts)
                        s2s_attn = s2s_attn.transpose(-1, -2)
                        s2s_attn = s2s_attn[..., 1:]
                        s2s_attn = s2s_attn.transpose(-1, -2)
                        mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                        s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                        t_en = model.text_encoder(texts, input_lengths, text_mask)
                        asr = (t_en @ s2s_attn_mono)
                        d_gt = s2s_attn_mono.sum(axis=-1).detach()

                    # Prosodic style for predictor (style_dim only; no acoustic style_encoder)
                    ss = []
                    for bib in range(len(mel_input_length)):
                        mel = mels[bib, :, :mel_input_length[bib]]
                        s = model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))
                        ss.append(s)
                    s = torch.stack(ss).squeeze()

                    bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
                    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
                    d, p = model.predictor(d_en, s, input_lengths, s2s_attn_mono, text_mask)

                    mel_len = int(mel_input_length.min().item() / 2 - 1)
                    en, gt, p_en, wav, s2s, uv = [], [], [], [], [], []

                    for bib in range(len(mel_input_length)):
                        mel_length = int(mel_input_length[bib].item() / 2)
                        random_start = np.random.randint(0, mel_length - mel_len)
                        en.append(asr[bib, :, random_start:random_start + mel_len])
                        s2s.append(s2s_attn_mono[bib, :, random_start:random_start + mel_len])
                        p_en.append(p[bib, :, random_start:random_start + mel_len])
                        gt.append(mels[bib, :, (random_start * 2):((random_start + mel_len) * 2)])
                        y = waves[bib][(random_start * 2) * 300:((random_start + mel_len) * 2) * 300]
                        wav.append(torch.from_numpy(y).to(device))

                    en = torch.stack(en)
                    p_en = torch.stack(p_en)
                    gt = torch.stack(gt).detach()
                    s2s = torch.stack(s2s).detach()

                    # Prosodic style for predictor (style_encoder not used for decoder c)
                    s = model.predictor_encoder(gt.unsqueeze(1))

                    F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1))
                    N_real = log_norm(gt.unsqueeze(1)).squeeze(1)
                    if "bertfusion" in cond_prosody_type:
                        cut_phn_start, cut_phn_end, _, _ = get_phone_range_by_cut_f2p_attn(s2s)
                        s2s_cutphone = s2s[:, cut_phn_start:cut_phn_end, :]
                        uv_masks_cut = uv_masks[:, cut_phn_start:cut_phn_end]
                        uv_mask_phn_ref, s2s_ref = uv_masks_cut.clone(), s2s_cutphone.clone()
                        trd = model.predictor.trd_encoding(F0_real, s2s_cutphone, uv_masks_cut, uv_mask_phn_ref, s2s_ref, drop_trend=False)
                        p_en = torch.cat([p_en, trd], dim=1)

                    F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s)

                    loss_dur = 0
                    for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                        _s2s_pred = _s2s_pred[:_text_length, :]
                        _text_input = _text_input[:_text_length].long()
                        _s2s_trg = torch.zeros_like(_s2s_pred)
                        for bib in range(_s2s_trg.shape[0]):
                            _s2s_trg[bib, :_text_input[bib]] = 1
                        _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)
                        loss_dur += F.l1_loss(_dur_pred[1:_text_length - 1],
                                              _text_input[1:_text_length - 1])
                    loss_dur /= texts.size(0)

                    pe = torch.cat([N_fake.unsqueeze(1), F0_fake.unsqueeze(1)], dim=1)

                    # CFMDecoderV4: no global c, no mask/p_mask args
                    loss_cfm, _ = model.decoder.compute_loss(gt, mu=en, seq_style=pe)

                    F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1))
                    loss_F0 = F.l1_loss(F0_real, F0_fake) / 10

                    loss_test += (loss_cfm).mean()
                    loss_align += (loss_dur).mean()
                    loss_f += (loss_F0).mean()

                    iters_test += 1
                except Exception as e:
                    print(f"run into exception", e)
                    traceback.print_exc()
                    continue

        print('Epochs:', epoch + 1)
        logger.info(
            'Validation loss: %.3f, Dur loss: %.3f, F0 loss: %.3f' % (loss_test / iters_test, loss_align / iters_test,
                                                                      loss_f / iters_test) + '\n\n\n')
        print('\n\n\n')
        writer.add_scalar('eval/mel_loss', loss_test / iters_test, epoch + 1)
        writer.add_scalar('eval/dur_loss', loss_align / iters_test, epoch + 1)
        writer.add_scalar('eval/F0_loss', loss_f / iters_test, epoch + 1)

        generator = get_vocoder(ckpt_dir="/home/rosen/ckpt/styletts/Vocoder/LibriTTS/", device=device)
        if epoch < joint_epoch:
            # ICL reconstruction examples with GT duration
            with torch.no_grad():
                for bib in range(len(asr)):
                    mel_length = int(mel_input_length[bib].item())
                    if mel_length < 10:
                        continue

                    # 30/70 split: tgt = first 70%, ref = next 30%
                    # Aligns inference mask ratio with training (mask_ratio_min=0.7)
                    ref_len = (mel_length * 3 // 10 // 2) * 2   # 30% of total, rounded to even
                    tgt_len = (mel_length * 7 // 10 // 2) * 2   # 70% of total, rounded to even
                    asr_ref = ref_len // 2
                    asr_tgt = tgt_len // 2

                    tgt_mel      = mels[bib, :, :tgt_len].unsqueeze(0)                       # [1, 80, tgt_len]
                    ref_mel_icl  = mels[bib, :, tgt_len:tgt_len + ref_len].unsqueeze(0)      # [1, 80, ref_len]
                    en_tgt       = asr[bib, :, :asr_tgt].unsqueeze(0)                        # [1, dim, asr_tgt]
                    en_ref       = asr[bib, :, asr_tgt:asr_tgt + asr_ref].unsqueeze(0)       # [1, dim, asr_ref]
                    s2s_slice_tgt = s2s_attn_mono[bib, :, :asr_tgt].unsqueeze(0)

                    # Reference pitch/energy (ground truth)
                    F0_ref, _, _ = model.pitch_extractor(ref_mel_icl.unsqueeze(1))
                    norm_ref = log_norm(ref_mel_icl.unsqueeze(1)).squeeze(1)
                    pe_ref = torch.cat([norm_ref.unsqueeze(1), F0_ref.unsqueeze(1)], dim=1)

                    # Target: GT pitch/energy for reconstruction
                    F0_tgt, _, _ = model.pitch_extractor(tgt_mel.unsqueeze(1))
                    norm_tgt = log_norm(tgt_mel.unsqueeze(1)).squeeze(1)
                    pe_tgt_gt = torch.cat([norm_tgt.unsqueeze(1), F0_tgt.unsqueeze(1)], dim=1)

                    # Reconstruction with GT pe (ICL)
                    mel_rec, _ = model.decoder(
                        mu_tgt=en_tgt,
                        n_timesteps=200,
                        temperature=1.0,
                        cond_ref=ref_mel_icl,
                        mu_ref=en_ref,
                        seq_style_tgt=pe_tgt_gt,
                        seq_style_ref=pe_ref,
                        cfg_strength=cfg_strength_val,
                        return_attn_map=False)

                    c = mel_rec.squeeze()
                    y_g_hat = generator(c.unsqueeze(0))
                    writer.add_audio('eval/y' + str(bib), y_g_hat.cpu().numpy().squeeze(), epoch, sample_rate=sr)

                    # Reconstruction with predictor pe (ICL)
                    s_dur_tgt = model.predictor_encoder(tgt_mel.unsqueeze(1))
                    p_en_tgt = p[bib, :, :asr_tgt].unsqueeze(0)

                    if "bertfusion" in cond_prosody_type:
                        cut_phn_start, cut_phn_end, _, _ = get_phone_range_by_cut_f2p_attn(s2s_slice_tgt.unsqueeze(0))
                        s2s_cutphone = s2s_slice_tgt[:, cut_phn_start:cut_phn_end, :]
                        uv_masks_cut = uv_masks[bib].unsqueeze(0)[:, cut_phn_start:cut_phn_end]
                        uv_masks_cut_tgt, s2s_cutphone_tgt = uv_masks_cut.clone(), s2s_cutphone.clone()
                        trd = model.predictor.trd_encoding(F0_tgt, s2s_cutphone, uv_masks_cut,
                                                           uv_masks_cut_tgt, s2s_cutphone_tgt, drop_trend=False)
                        p_en_tgt = torch.cat([p_en_tgt, trd], dim=1)

                    F0_fake_tgt, N_fake_tgt = model.predictor.F0Ntrain(p_en_tgt, s_dur_tgt)
                    pe_tgt_pred = torch.cat([N_fake_tgt.unsqueeze(1), F0_fake_tgt.unsqueeze(1)], dim=1)

                    mel_pred, _ = model.decoder(
                        mu_tgt=en_tgt,
                        n_timesteps=200,
                        temperature=1.0,
                        cond_ref=ref_mel_icl,
                        mu_ref=en_ref,
                        seq_style_tgt=pe_tgt_pred,
                        seq_style_ref=pe_ref,
                        cfg_strength=cfg_strength_val,
                        return_attn_map=False)

                    c_pred = mel_pred.squeeze()
                    y_pred = generator(c_pred.unsqueeze(0))
                    writer.add_audio('pred/y' + str(bib), y_pred.cpu().numpy().squeeze(), epoch, sample_rate=sr)

                    if epoch == 0:
                        writer.add_audio('gt/y' + str(bib), waves[bib].squeeze(), epoch, sample_rate=sr)

                    if bib >= 5:
                        break
        else:
            # ICL TTS from text with diffusion-predicted style
            with torch.no_grad():
                if multispeaker and epoch >= diff_epoch:
                    ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))

                for bib in range(len(d_en)):
                    if multispeaker:
                        s_pred = sampler(noise=torch.randn((1, style_dim)).unsqueeze(1).to(texts.device),
                                         embedding=bert_dur[bib].unsqueeze(0),
                                         embedding_scale=1,
                                         features=ref_sp[bib].unsqueeze(0),
                                         num_steps=5).squeeze(1)
                    else:
                        s_pred = sampler(noise=torch.randn((1, style_dim)).unsqueeze(1).to(texts.device),
                                         embedding=bert_dur[bib].unsqueeze(0),
                                         embedding_scale=1,
                                         num_steps=5).squeeze(1)

                    # s_pred is prosodic style only (style_dim)
                    s = s_pred

                    d = model.predictor.text_encoder(d_en[bib, :, :input_lengths[bib]].unsqueeze(0),
                                                     s, input_lengths[bib, ...].unsqueeze(0),
                                                     text_mask[bib, :input_lengths[bib]].unsqueeze(0))
                    x, _ = model.predictor.lstm(d)
                    duration = model.predictor.duration_proj(x)
                    duration = torch.sigmoid(duration).sum(axis=-1)
                    pred_dur = torch.round(duration.squeeze()).clamp(min=1)
                    pred_dur[-1] += 5

                    pred_aln_trg = torch.zeros(input_lengths[bib], int(pred_dur.sum().data))
                    c_frame = 0
                    for ii in range(pred_aln_trg.size(0)):
                        pred_aln_trg[ii, c_frame:c_frame + int(pred_dur[ii].data)] = 1
                        c_frame += int(pred_dur[ii].data)

                    en_tts = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(texts.device))
                    if "bertfusion" in cond_prosody_type:
                        uv_masks_cut_tgt = uv_masks[bib].unsqueeze(0).clone()
                        trd = model.predictor.trd_encoding(F0_real[bib][None, ...], s2s[bib].unsqueeze(0),
                                                           uv_masks[bib].unsqueeze(0), uv_masks_cut_tgt,
                                                           pred_aln_trg.unsqueeze(0).to(texts.device),
                                                           drop_trend=False)
                        en_tts = torch.cat([en_tts, trd], dim=1)

                    F0_pred, N_pred = model.predictor.F0Ntrain(en_tts, s)
                    pe_tts = torch.cat([N_pred.unsqueeze(1), F0_pred.unsqueeze(1)], dim=1)

                    # mu_tgt for decoder = t_en @ pred_aln (without trd, matching original)
                    mu_tgt_tts = (t_en[bib, :, :input_lengths[bib]].unsqueeze(0)
                                  @ pred_aln_trg.unsqueeze(0).to(texts.device))

                    # Reference: use last 30% of ground-truth mel for ICL conditioning
                    mel_length   = int(mel_input_length[bib].item())
                    ref_len_tts  = (mel_length * 3 // 10 // 2) * 2   # 30%, rounded to even
                    asr_ref_tts  = ref_len_tts // 2
                    ref_start_mel = mel_length - ref_len_tts
                    ref_start_asr = ref_start_mel // 2
                    ref_mel_icl = mels[bib, :, ref_start_mel:ref_start_mel + ref_len_tts].unsqueeze(0)
                    en_ref_icl  = asr[bib, :, ref_start_asr:ref_start_asr + asr_ref_tts].unsqueeze(0)
                    F0_ref_tts, _, _ = model.pitch_extractor(ref_mel_icl.unsqueeze(1))
                    norm_ref_tts = log_norm(ref_mel_icl.unsqueeze(1)).squeeze(1)
                    pe_ref_tts = torch.cat([norm_ref_tts.unsqueeze(1), F0_ref_tts.unsqueeze(1)], dim=1)

                    out, _ = model.decoder(
                        mu_tgt=mu_tgt_tts,
                        n_timesteps=200,
                        temperature=1.0,
                        cond_ref=ref_mel_icl,
                        mu_ref=en_ref_icl,
                        seq_style_tgt=pe_tts,
                        seq_style_ref=pe_ref_tts,
                        cfg_strength=cfg_strength_val,
                        return_attn_map=False)

                    out = out.squeeze()
                    out = generator(out.unsqueeze(0))
                    writer.add_audio('pred/y' + str(bib), out.cpu().numpy().squeeze(), epoch, sample_rate=sr)
                    if bib >= 5:
                        break

        if epoch % saving_epoch == 0 and epoch != start_epoch:
            if (loss_test / iters_test) < best_loss:
                best_loss = loss_test / iters_test
            print('Saving..')
            state = {
                'net': {key: model[key].state_dict() for key in model},
                'optimizer': optimizer.state_dict(),
                'iters': iters,
                'val_loss': loss_test / iters_test,
                'epoch': epoch}
            save_path = osp.join(log_dir, 'epoch_2nd_%05d.pth' % epoch)
            torch.save(state, save_path)

            if model_params.diffusion.dist.estimate_sigma_data:
                config['model_params']['diffusion']['dist']['sigma_data'] = float(np.mean(running_std))
                with open(osp.join(log_dir, osp.basename(config_path)), 'w') as outfile:
                    yaml.dump(config, outfile, default_flow_style=True)
            if model_params.stats_pe:
                config['model_params']['pe_min_max'] = [
                float(pitch_min),
                float(pitch_max),
                float(energy_min),
                float(energy_max)]
                with open(osp.join(log_dir, osp.basename(config_path)), 'w') as outfile:
                    yaml.dump(config, outfile, default_flow_style=True)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def load_checkpoint_vocoder(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_vocoder(ckpt_dir="/home/rosen/ckpt/styletts/Vocoder/LibriTTS/", device=None):
    cp_g = scan_checkpoint(ckpt_dir, 'g_')
    config_file = os.path.join(os.path.split(cp_g)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint_vocoder(cp_g, device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()
    return generator


def log_grad_params(model):
    total_norm = {}
    for key in model.keys():
        total_norm[key] = 0
        parameters = [p for p in model[key].parameters() if p.grad is not None and p.requires_grad]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm[key] += param_norm.item() ** 2
        total_norm[key] = total_norm[key] ** 0.5
    return total_norm


if __name__ == "__main__":
    main()
