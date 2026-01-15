# load packages
import random
import time
import click
import shutil
import traceback
import warnings

import torch

warnings.simplefilter('ignore')
from torch.utils.tensorboard import SummaryWriter

from meldataset import build_dataloader
from Utils.PLBERT.util import load_plbert

from models_txt2mel_cfm import *
from losses import *
from utils import *
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from optimizers import build_optimizer
from attrdict import AttrDict
from Modules.hifi_gan.vocoder import Generator
import glob
from utils import r1_reg, adv_loss
import json
from utilities.guide_mask import make_guided_attention_masks2

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
@click.option('-p', '--config_path', default='Configs/config_libritts_txt2mel_cfm_v19.yml', type=str)
def main(config_path):
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
        train_list = train_list[:int(len(train_list) * data_ratio)] # to save time

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
    ## styleTTS2 param
    model_params = recursive_munch(config['model_params'])
    multispeaker = model_params.multispeaker
    learn_monoAttn = model_params.learn_monoAttn
    multiply_mono = model_params.get("multiply_mono", False)
    cond_prosody_type = model_params.get("cond_prosody_type", "predict")
    ## cfm param
    cfm_params = recursive_munch(config['cfm_config'])
    pitch_min, pitch_max, energy_min, energy_max = tuple(cfm_params.pe_min_max)


    model = build_model(model_params, config['cfm_config'], text_aligner, pitch_extractor, plbert)
    cfg_dropout = config['cfm_config'].get("cfg_dropout", 0)
    style_dim = model_params.style_dim

    _ = [model[key].to(device) for key in model]

    start_epoch = 0
    load_pretrained = config.get('pretrained_model', '') != '' and config.get('second_stage_load_pretrained', False)

    # load first
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
                                                                           'diffusion'])  # keep starting epoch for tensorboard log
            # these epochs should be counted from the start epoch
            diff_epoch += start_epoch
            joint_epoch += start_epoch
            epochs += start_epoch
            model.predictor_encoder = copy.deepcopy(model.style_encoder)
        else:
            raise ValueError('You need to specify the path to the first stage model.')

    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),  # empirical parameters
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

    # load second models if there is a model
    if load_pretrained:
        model, optimizer, start_epoch, iters = load_checkpoint(model, optimizer, config['pretrained_model'],
                                                               load_only_params=config.get('load_only_params', True))

    n_down = model.text_aligner.n_down

    best_loss = float('inf')  # best test loss
    iters = 0

    criterion = nn.L1Loss()  # F0 loss (regression)
    torch.cuda.empty_cache()
    print('BERT', optimizer.optimizers['bert'])
    print('decoder', optimizer.optimizers['decoder'])

    start_ds = False
    running_std = []
    VIS_PSD_CONTOUR = True
    for epoch in range(start_epoch, epochs):
        running_loss = 0
        start_time = time.time()

        _ = [model[key].eval() for key in model]

        model.predictor.train()
        model.bert_encoder.train()
        model.bert.train()

        if epoch >= diff_epoch:
            start_ds = True

        for i, batch in enumerate(train_dataloader):
            waves = batch[0]
            batch = [b.to(device) for b in batch[1:]]
            texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch

            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                mel_mask = length_to_mask(mel_input_length).to(device)
                text_mask = length_to_mask(input_lengths).to(texts.device)
                # GD txt/mel attention for duration
                try:
                    _, _, s2s_attn = model.text_aligner(mels, mask, texts)
                    s2s_attn = s2s_attn.transpose(-1, -2)
                    s2s_attn = s2s_attn[..., 1:]
                    s2s_attn = s2s_attn.transpose(-1, -2)
                except:
                    continue
                mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                # acoustic text embedding (t_en) and its gd_dur-extended version (asr)
                t_en = model.text_encoder(texts, input_lengths, text_mask)
                asr = (t_en @ s2s_attn_mono)
                d_gt = s2s_attn_mono.sum(axis=-1).detach()

                # Get reference styles
                if multispeaker and epoch >= diff_epoch:
                    ref_ss = model.style_encoder(ref_mels.unsqueeze(1))
                    ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))
                    ref = torch.cat([ref_ss, ref_sp], dim=1)

            # GD acoustic (gs) / prosodic (ss) style of the entire utterance for style diffuser training
            ss, gs = [], []
            for bib in range(len(mel_input_length)): # this operation cannot be done in batch because of the avgpool layer (may need to work on masked avgpool)
                mel_length = int(mel_input_length[bib].item())
                mel = mels[bib, :, :mel_input_length[bib]]
                s = model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))
                ss.append(s)
                s = model.style_encoder(mel.unsqueeze(0).unsqueeze(1))
                gs.append(s)
            s_dur = torch.stack(ss).squeeze()  # global prosodic styles
            gs = torch.stack(gs).squeeze()  # global acoustic styles
            s_trg = torch.cat([gs, s_dur], dim=-1).detach()  # ground truth for denoiser

            # Predict acoustic / prosodic (s_preds) style by denoiser given semantic embedding (bert) of text
            bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
            ## Train denoiser
            if epoch >= diff_epoch:
                num_steps = np.random.randint(3, 5)
                if model_params.diffusion.dist.estimate_sigma_data:
                    model.diffusion.diffusion.sigma_data = s_trg.std(axis=-1).mean().item()  # batch-wise std estimation
                    running_std.append(model.diffusion.diffusion.sigma_data)
                if multispeaker:
                    s_preds = sampler(noise=torch.randn_like(s_trg).unsqueeze(1).to(device),
                                      embedding=bert_dur,
                                      embedding_scale=1,
                                      features=ref,  # reference from the same speaker as the embedding
                                      embedding_mask_proba=0.1,
                                      num_steps=num_steps).squeeze(1)
                    loss_diff = model.diffusion(s_trg.unsqueeze(1), embedding=bert_dur, features=ref).mean()  # EDM loss
                    loss_sty = F.l1_loss(s_preds, s_trg.detach())  # style reconstruction loss
                    #s_trg_norm, s_preds_norm = torch.norm(s_trg), torch.norm(s_preds)
                    #print("s_trg_norm, s_preds_norm, loss_sty: ", s_trg_norm.item(), s_preds_norm.item(), loss_sty.item())
                else:
                    s_preds = sampler(noise=torch.randn_like(s_trg).unsqueeze(1).to(device),
                                      embedding=bert_dur,
                                      embedding_scale=1,
                                      embedding_mask_proba=0.1,
                                      num_steps=num_steps).squeeze(1)
                    loss_diff = model.diffusion.diffusion(s_trg.unsqueeze(1), embedding=bert_dur).mean()  # EDM loss
                    loss_sty = F.l1_loss(s_preds, s_trg.detach())  # style reconstruction loss
            else:
                loss_sty = 0
                loss_diff = 0

            # Predicted dur (d) and gd_dur-extended prosody-predicted embedding (p) given semantic embedding (d_en) and prosodic style (s_dur)
            d, p = model.predictor(d_en, s_dur, input_lengths, s2s_attn_mono, text_mask)
            mel_len = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)
            mel_len_st = int(mel_input_length.min().item() / 2 - 1)
            en, gt, st, p_en, wav, s2s = [], [], [], [], [], []   # en=asr, gt=mels, st=mels_v2, p_en=p, wav

            # Clip the mel, asr (text), p (prosody), mel_v2 (style) for training
            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)
                random_start = np.random.randint(0, mel_length - mel_len)
                en.append(asr[bib, :, random_start:random_start + mel_len])
                s2s.append(s2s_attn_mono[bib, :, random_start:random_start + mel_len])
                p_en.append(p[bib, :, random_start:random_start + mel_len])
                gt.append(mels[bib, :, (random_start * 2):((random_start + mel_len) * 2)])
                y = waves[bib][(random_start * 2) * 300:((random_start + mel_len) * 2) * 300]
                wav.append(torch.from_numpy(y).to(device))

                # style reference (better to be different from the GT)
                random_start = np.random.randint(0, mel_length - mel_len_st)
                st.append(mels[bib, :, (random_start * 2):((random_start + mel_len_st) * 2)])
            # mask of clipped mel
            with torch.no_grad():
                mel_length_cut = torch.where(mel_input_length < mel_len * 2, mel_input_length, torch.tensor(mel_len * 2))
                mel_cut_mask = length_to_mask(mel_length_cut).to('cuda')

            en, p_en, gt, st, s2s = torch.stack(en), torch.stack(p_en), torch.stack(gt).detach(), torch.stack(st).detach(), torch.stack(s2s).detach()
            if gt.size(-1) < 80:
                continue

            # GD acoustic (s) / prosodic (s_dur) style given clipped (!not Entire) mel for pe prediction
            s_dur = model.predictor_encoder(st.unsqueeze(1) if multispeaker else gt.unsqueeze(1))  # *
            s = model.style_encoder(st.unsqueeze(1) if multispeaker else gt.unsqueeze(1))

            # GD, predicted pe, and its loss
            with torch.no_grad():
                F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1))
                N_real = log_norm(gt.unsqueeze(1)).squeeze(1)
            F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s_dur)
            loss_F0_rec = (F.smooth_l1_loss(F0_real, F0_fake)) / 10
            loss_norm_rec = F.smooth_l1_loss(N_real, N_fake)
            # update F0, N stats
            pitch_min, pitch_max = min(pitch_min, torch.min(F0_real).item()), max(pitch_max, torch.max(F0_real).item())
            energy_min, energy_max = min(energy_min, torch.min(N_real).item()), max(energy_max, torch.max(N_real).item())

            if cond_prosody_type == "hierstyle":
                pe = torch.cat([N_fake.unsqueeze(1), F0_fake.unsqueeze(1)], dim=1)
                s2s = F.interpolate(s2s, size=F0_real.size(-1), mode="nearest")
                _, _, N_real_phone, F0_real_phone = frame_to_phoneme_avg_and_back_binary(N_real, F0_real, s2s)
                pe_real = torch.cat([N_real_phone.unsqueeze(1), F0_real_phone.unsqueeze(1)], dim=1)

                # dur alignment of pe_real to pe given pe_voice_mask
                pe_voice_mask = (pe[:, 1, :] >= 50.0) & (pe[:, 1, :] <= 600.0) & torch.isfinite(pe[:, 1, :])
                pe_real_ori = pe_real.clone()
                pe_real = align_dur2(pe_real, pe_voice_mask.unsqueeze(1), pitch_idx=1)

                # VIS difference of pe_real, pe, and dur_aligned_pe
                """
                if VIS_PSD_CONTOUR:
                    from exp.vis2 import plot_f0_comparison
                    for i in range(5):
                        for j in range(2):
                            psd_type = ["pitch", "energy"]
                            plot_f0_comparison(pe_real_ori[i, j], pe[i, j], pe_real[i, j],
                                               out_path=f"res/temp/{psd_type[j]}_real_pred_realAligned_{i}_epoch{epoch}.png",
                                               labels=(f"{psd_type[j]}_real", f"{psd_type[j]}_pred", f"{psd_type[j]}_dur_aligned"))
                            plot_f0_comparison(pe_real_ori[i, j], pe_voice_mask[i] * 20, pe_real[i, j],
                                               out_path=f"res/temp/{psd_type[j]}_real_predmask_realAligned_{i}_epoch{epoch}.png",
                                               labels=(f"{psd_type[j]}_real", f"{psd_type[j]}_pred_mask", f"{psd_type[j]}_dur_aligned"))
                    VIS_PSD_CONTOUR = False
                """
            else:
                pe = torch.cat([N_fake.unsqueeze(1), F0_fake.unsqueeze(1)], dim=1)
                pe_real = None

            # Start training: CFM_loss, dur loss (dur, ce)
            optimizer.zero_grad()
            loss_cfm, attn_maps = model.decoder.compute_loss(gt, ~mel_cut_mask.unsqueeze(1), mu=en, c=s,
                                                             seq_style=pe, p_mask=~mel_cut_mask.unsqueeze(1), seq_style_gt=pe_real)
            loss_ce, loss_dur = 0, 0
            ## Predicted (d) and gd (d_gt) dur
            for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                _s2s_pred = _s2s_pred[:_text_length, :]
                _text_input = _text_input[:_text_length].long()
                _s2s_trg = torch.zeros_like(_s2s_pred)
                for p in range(_s2s_trg.shape[0]):
                    _s2s_trg[p, :_text_input[p]] = 1
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
                #from IPython.core.debugger import set_trace
                #set_trace()
            optimizer.step('bert_encoder')
            optimizer.step('bert')
            optimizer.step('predictor')
            optimizer.step('predictor_encoder')

            # training stage 2 and 3
            if epoch >= diff_epoch:
                optimizer.step('diffusion')
            if epoch >= joint_epoch:  #(joint_epoch > diff_epoch)
                optimizer.step('decoder')
            iters = iters + 1

            # print out main mIndex
            if (i + 1) % log_interval == 0:
                logger.info(
                    'Epoch [%d/%d], Step [%d/%d], Loss: %.5f, cfm Loss: %.5f, Dur Loss: %.5f, CE Loss: %.5f, Norm Loss: %.5f, F0 Loss: %.5f, Sty Loss: %.5f, Diff Loss: %.5f'
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

                # print grad
                #total_norm = log_grad_params(model)
                #logger.info(str(total_norm).replace("\n", ""))
                print('Time elasped:', time.time() - start_time)

        loss_test = 0
        loss_align = 0
        loss_f = 0
        _ = [model[key].eval() for key in model]

        # save pitch_min/max, energy_min/max
        if epoch == start_epoch:
            with open(osp.join(log_dir, "pe_stats.txt"), 'w') as outfile:
                outfile.write(",".join(map(str, [pitch_min, pitch_max, energy_min, energy_max]))) # copy this value to pe_min_max in config_*.yml, and train again.

        # start evaluation loss, and predict demo speech with gd_dur and pred_dur
        with torch.no_grad():
            iters_test = 0
            for batch_idx, batch in enumerate(val_dataloader):
                optimizer.zero_grad()
                try:
                    waves = batch[0]
                    batch = [b.to(device) for b in batch[1:]]
                    texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch
                    with torch.no_grad():
                        mask = length_to_mask(mel_input_length // (2 ** n_down)).to('cuda')
                        text_mask = length_to_mask(input_lengths).to(texts.device)
                        _, _, s2s_attn = model.text_aligner(mels, mask, texts)
                        s2s_attn = s2s_attn.transpose(-1, -2)
                        s2s_attn = s2s_attn[..., 1:]
                        s2s_attn = s2s_attn.transpose(-1, -2)
                        mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                        s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                        # encode
                        t_en = model.text_encoder(texts, input_lengths, text_mask)
                        asr = (t_en @ s2s_attn_mono)
                        d_gt = s2s_attn_mono.sum(axis=-1).detach()

                    ss, gs = [], []
                    with torch.no_grad():  # get
                        mel_length_cut = torch.where(mel_input_length < mel_len * 2, mel_input_length,
                                                     torch.tensor(mel_len * 2))
                        mel_cut_mask = length_to_mask(mel_length_cut).to('cuda')

                    for bib in range(len(mel_input_length)):
                        mel_length = int(mel_input_length[bib].item())
                        mel = mels[bib, :, :mel_input_length[bib]]
                        s = model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))
                        ss.append(s)
                        s = model.style_encoder(mel.unsqueeze(0).unsqueeze(1))
                        gs.append(s)

                    s = torch.stack(ss).squeeze()
                    gs = torch.stack(gs).squeeze()
                    s_trg = torch.cat([s, gs], dim=-1).detach()

                    bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
                    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
                    d, p = model.predictor(d_en, s, input_lengths, s2s_attn_mono, text_mask)
                    # get clips
                    mel_len = int(mel_input_length.min().item() / 2 - 1)
                    en, gt, p_en, wav, s2s = [], [], [], [], []

                    for bib in range(len(mel_input_length)):
                        mel_length = int(mel_input_length[bib].item() / 2)
                        random_start = np.random.randint(0, mel_length - mel_len)
                        en.append(asr[bib, :, random_start:random_start + mel_len])
                        s2s.append(s2s_attn_mono[bib, :, random_start:random_start + mel_len])
                        p_en.append(p[bib, :, random_start:random_start + mel_len])
                        gt.append(mels[bib, :, (random_start * 2):((random_start + mel_len) * 2)])
                        y = waves[bib][(random_start * 2) * 300:((random_start + mel_len) * 2) * 300]
                        wav.append(torch.from_numpy(y).to(device))

                    wav = torch.stack(wav).float().detach()
                    en = torch.stack(en)
                    p_en = torch.stack(p_en)
                    gt = torch.stack(gt).detach()
                    s2s = torch.stack(s2s).detach()

                    s = model.predictor_encoder(gt.unsqueeze(1))
                    F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s)

                    # gt F0, energy
                    F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1))
                    N_real = log_norm(gt.unsqueeze(1)).squeeze(1)

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
                    s = model.style_encoder(gt.unsqueeze(1))

                    if cond_prosody_type == "hierstyle":
                        pe = torch.cat([N_fake.unsqueeze(1), F0_fake.unsqueeze(1)], dim=1)
                        s2s = F.interpolate(s2s, size=F0_real.size(-1), mode="nearest")
                        _, _, N_real_phone, F0_real_phone = frame_to_phoneme_avg_and_back_binary(N_real, F0_real, s2s)
                        pe_real = torch.cat([N_real_phone.unsqueeze(1), F0_real_phone.unsqueeze(1)], dim=1)

                        # dur alignment of pe_real to pe given pe_voice_mask
                        pe_voice_mask = (pe[:, 1, :] >= 50.0) & (pe[:, 1, :] <= 600.0) & torch.isfinite(pe[:, 1, :])
                        pe_real = align_dur2(pe_real, pe_voice_mask.unsqueeze(1), pitch_idx=1)
                    else:
                        pe = torch.cat([N_fake.unsqueeze(1), F0_fake.unsqueeze(1)], dim=1)
                        pe_real = None

                    loss_cfm, attn_maps = model.decoder.compute_loss(
                        gt, ~mel_cut_mask.unsqueeze(1), mu=en, c=s, seq_style=pe, p_mask=~mel_cut_mask.unsqueeze(1), seq_style_gt=pe_real)

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
            # generating reconstruction examples with GT duration
            with torch.no_grad():
                for bib in range(len(asr)):
                    mel_length = int(mel_input_length[bib].item())
                    gt = mels[bib, :, :mel_length].unsqueeze(0)
                    en = asr[bib, :, :mel_length // 2].unsqueeze(0)
                    s2s_slice = s2s_attn_mono[bib, :, :mel_length].unsqueeze(0)

                    F0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))
                    #F0_real = F0_real.unsqueeze(0)
                    s = model.style_encoder(gt.unsqueeze(1))
                    real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)

                    #pe = torch.cat([real_norm.unsqueeze(1), F0_real.unsqueeze(1)], dim=1)
                    if cond_prosody_type == "hierstyle":
                        pe = torch.cat([real_norm.unsqueeze(1), F0_real.unsqueeze(1)], dim=1)
                        s2s_slice = F.interpolate(s2s_slice, size=F0_real.size(-1), mode="nearest")
                        _, _, N_real_phone, F0_real_phone = frame_to_phoneme_avg_and_back_binary(real_norm, F0_real, s2s_slice)
                        pe_real = torch.cat([N_real_phone.unsqueeze(1), F0_real_phone.unsqueeze(1)], dim=1)

                        # dur alignment of pe_real to pe given pe_voice_mask
                        pe_voice_mask = (pe[:, 1, :] >= 50.0) & (pe[:, 1, :] <= 600.0) & torch.isfinite(pe[:, 1, :])
                        pe_real = align_dur2(pe_real, pe_voice_mask.unsqueeze(1), pitch_idx=1)
                    else:
                        pe = torch.cat([real_norm.unsqueeze(1), F0_real.unsqueeze(1)], dim=1)
                        pe_real = None

                    cfg_strength = 3 if cfg_dropout > 0 else None
                    mel_rec, _ = model.decoder(mu=en, mask=mask, n_timesteps=200, temperature=1.0, c=s, seq_style=pe, p_mask=mask,
                                               cfg_strength=cfg_strength, seq_style_gt=pe_real)

                    # add vocoder
                    c = mel_rec.squeeze()
                    y_g_hat = generator(c.unsqueeze(0))

                    writer.add_audio('eval/y' + str(bib), y_g_hat.cpu().numpy().squeeze(), epoch, sample_rate=sr)

                    s_dur = model.predictor_encoder(gt.unsqueeze(1))
                    p_en = p[bib, :, :mel_length // 2].unsqueeze(0)

                    F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s_dur)

                    #pe = torch.cat([N_fake.unsqueeze(1), F0_fake.unsqueeze(1)], dim=1)
                    if cond_prosody_type == "hierstyle":
                        pe = torch.cat([N_fake.unsqueeze(1), F0_fake.unsqueeze(1)], dim=1)
                        s2s_slice = F.interpolate(s2s_slice, size=F0_real.size(-1), mode="nearest")
                        _, _, N_real_phone, F0_real_phone = frame_to_phoneme_avg_and_back_binary(real_norm, F0_real, s2s_slice)
                        pe_real = torch.cat([N_real_phone.unsqueeze(1), F0_real_phone.unsqueeze(1)], dim=1)

                        # dur alignment of pe_real to pe given pe_voice_mask
                        pe_voice_mask = (pe[:, 1, :] >= 50.0) & (pe[:, 1, :] <= 600.0) & torch.isfinite(pe[:, 1, :])
                        pe_real = align_dur2(pe_real, pe_voice_mask.unsqueeze(1), pitch_idx=1)
                    else:
                        pe = torch.cat([N_fake.unsqueeze(1), F0_fake.unsqueeze(1)], dim=1)
                        pe_real = None

                    cfg_strength = 3 if cfg_dropout > 0 else None
                    mel_pred, _ = model.decoder(mu=en, mask=mask, n_timesteps=200, temperature=1.0, c=s, seq_style=pe, p_mask=mask, cfg_strength=cfg_strength, seq_style_gt=pe_real)

                    # add vocoder
                    c_pred = mel_pred.squeeze()
                    y_pred = generator(c_pred.unsqueeze(0))

                    writer.add_audio('pred/y' + str(bib), y_pred.cpu().numpy().squeeze(), epoch, sample_rate=sr)  # why only show epoch14?

                    if epoch == 0:
                        writer.add_audio('gt/y' + str(bib), waves[bib].squeeze(), epoch, sample_rate=sr)

                    if bib >= 5:
                        break
        else:
            # generating sampled speech from text directly
            with torch.no_grad():
                # compute reference styles
                if multispeaker and epoch >= diff_epoch:
                    ref_ss = model.style_encoder(ref_mels.unsqueeze(1))
                    ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))
                    ref_s = torch.cat([ref_ss, ref_sp], dim=1)

                for bib in range(len(d_en)):
                    if multispeaker:
                        s_pred = sampler(noise=torch.randn((1, style_dim * 2)).unsqueeze(1).to(texts.device),
                                         embedding=bert_dur[bib].unsqueeze(0),
                                         embedding_scale=1,
                                         features=ref_s[bib].unsqueeze(0),
                                         # reference from the same speaker as the embedding
                                         num_steps=5).squeeze(1)
                    else:
                        s_pred = sampler(noise=torch.randn((1, style_dim * 2)).unsqueeze(1).to(texts.device),
                                         embedding=bert_dur[bib].unsqueeze(0),
                                         embedding_scale=1,
                                         num_steps=5).squeeze(1)
                    ref = s_pred[:, :style_dim]
                    s = s_pred[:, style_dim:]

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
                    for i in range(pred_aln_trg.size(0)):
                        pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                        c_frame += int(pred_dur[i].data)

                    # encode prosody
                    en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(texts.device))
                    F0_pred, N_pred = model.predictor.F0Ntrain(en, s) # (1, T)
                    cfg_strength = 3 if cfg_dropout > 0 else None

                    if cond_prosody_type == "hierstyle":
                        pe = torch.cat([N_pred.unsqueeze(1), F0_pred.unsqueeze(1)], dim=1)
                        s2s = F.interpolate(s2s, size=F0_real.size(-1), mode="nearest")
                        _, _, N_real_phone, F0_real_phone = frame_to_phoneme_avg_and_back_binary(N_real, F0_real, s2s)
                        pe_real = torch.cat([N_real_phone[bib][None, None, ...], F0_real_phone[bib][None, None, ...]], dim=1)

                        # dur alignment of pe_real to pe given pe_voice_mask
                        pe_voice_mask = (pe[:, 1, :] >= 50.0) & (pe[:, 1, :] <= 600.0) & torch.isfinite(pe[:, 1, :])
                        pe_real_ori = pe_real.clone()
                        pe_real = align_dur2(pe_real, pe_voice_mask.unsqueeze(1), pitch_idx=1)

                        # VIS difference of pe_real, pe, and dur_aligned_pe
                        if VIS_PSD_CONTOUR:
                            from exp.vis2 import plot_f0_comparison
                            for i in range(1):
                                plot_f0_comparison(pe_real_ori[i, 1], pe[i, 1], pe_real[i, 1],
                                                   out_path=f"res/hierstyle_cond/pitch_real_pred_realAligned_{i}_epoch{epoch}.png",
                                                   labels=("pitch_real", "pitch_pred", "pitch_dur_aligned"))
                                plot_f0_comparison(pe_real_ori[i, 1], pe_voice_mask[i] * 20, pe_real[i, 1],
                                                   out_path=f"res/hierstyle_cond/pitch_real_predmask_realAligned_{i}_epoch{epoch}.png",
                                                   labels=("pitch_real", "pitch_pred_mask", "pitch_dur_aligned"))
                    else:
                        pe = torch.cat([N_pred.unsqueeze(1), F0_pred.unsqueeze(1)], dim=1)
                        pe_real = None

                    """
                    if cond_prosody_type == "hierstyle":
                        pe = torch.cat([N_real[bib].unsqueeze(0), F0_real[bib].unsqueeze(0)], dim=0).unsqueeze(0)
                    else:
                        pe = torch.cat([N_pred.unsqueeze(1), F0_pred.unsqueeze(1)], dim=1)  # N_pred: based on pred_dur by diffusion,  N_fake: based on gt_dur
                    """
                    out, _ = model.decoder(mu=t_en[bib, :, :input_lengths[bib]].unsqueeze(0) @ pred_aln_trg.unsqueeze(0).to(texts.device),
                                           mask=mask, n_timesteps=200, temperature=1.0, c=ref, seq_style=pe, p_mask=mask, cfg_strength=cfg_strength,
                                           seq_style_gt=pe_real)
                    # add vocoder
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

            # if estimate sigma, save the estimated simga
            if model_params.diffusion.dist.estimate_sigma_data:
                config['model_params']['diffusion']['dist']['sigma_data'] = float(np.mean(running_std))
                with open(osp.join(log_dir, osp.basename(config_path)), 'w') as outfile:
                    yaml.dump(config, outfile, default_flow_style=True)
            if model_params.stats_pe:
                config['cfm_config']['pe_min_max'] = [
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
    # load vocoder
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

def choose_mono_guide_delta():
    u = random.random()
    if u < 0.25:
        return None
    elif u < 0.5:
        return 0.2
    elif u < 0.75:
        return 0.5
    else:
        return 0.8

if __name__ == "__main__":
    main()