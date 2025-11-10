## StyleTTS2
### Train_first
- Load models, schedule
  - adjust bert, acoustic module lr
  - Load latest model
  - load ASR, pitch_extractor, plbert

- GET encoded text with text aligner
    - text_aligner(mel, text)        : s2s_pred, s2s_attn
    - maximum_path(s2s_attn, mask_ST): s2s_attn_mono
    - text_encoder(text):t_en
    - t_en @ s2s_attn_mono           : asr
    - s2s_attn_mono.sum(axis=-1)     : d_gt

- GET GD style/predict encoding from ref_mels (part?) and mels (whole)
  - style_encoder(ref_mels/mels)     : ref_ss/ss(or s_dur)
  - predictor_encoder(ref_mels/mels) : ref_sp/gs
  - concate(ref_ss/ss, ref_sp/gs)    : ref/s_trg

- GET bert-embeded text
  - bert(text)                       : bert_dur
  - bert_encoder(bert_dur)           : d_en

- GET GD pitch/energy/style
  - cut(asr)                                : en
  - cut(mels)                               : gt
  - cut(mels, 2len)                         : st
  - pitch_extractor(gt)                     : F0_real
  - style_encoder(gt)                       : s
  - log_norm(gt)                            : N_real

- [epoch>0] Train discriminator (start_ds)
  - decoder(en, F0_real, real_norm, s)      : mel_rec
  - discriminator(gt)                       : out
  - adv_loss(out, 1)                        : loss_real
  - r1_reg(out, gt)                         : loss_reg
  - discriminator(mel_rec)                  : out
  - adv_loss(out, 0)                        : loss_fake
  - loss_real + loss_fake + loss_reg        : d_loss
  - criterion(mel_rec, gt)        : loss_mel

- [epoch>0] Train dur
  - cross_entropy(s2s_pred, texts)          : loss_s2s
  - l1_loss(s2s_attn, s2s_attn_mono)        : loss_mono


- [epoch>0] Train generator
  - discriminator(mel_rec)                  : out_rec, f_fake
  - adv_loss(out_rec, 1)                    : loss_adv
  - model.discriminator(gt)                 : f_real 
  - mean(abs(f_real[m][k] - f_fake[m][k]))  : loss_fm

- [epoch>0]        Train dur, text 
  - torch.sigmoid(d)                        : _dur_pred
  - l1_loss(_dur_pred, d_gt)                : loss_dur
  - binary_cross_entropy_with_logits(d,d_gt): loss_ce

- Sum all loss
  - loss_mel + loss_adv + loss_fm + loss_mono + loss_s2s
  - running_loss += loss_mel 

- validate loss, synthesize speech(gd/pred)
  - mel_loss
  - dur_loss
  - F0_loss
  - model.decoder(en, F0_fake, N_fake, s)  : mel_pred
  - generator(mel_pred)                    : y_pred
  - writer.add_audio(y_pred)               : NONE       

- saving model

## Train_second flow
1. Load models, schedule
  - adjust bert, acoustic module lr
  - Load latest model
  - load ASR, pitch_extractor, plbert

2. GET encoded text with text aligner
    - text_aligner(mel, text):s2s_attn
    - maximum_path(s2s_attn, mask_ST): s2s_attn_mono
    - text_encoder(text):t_en
    - t_en @ s2s_attn_mono           : asr
    - s2s_attn_mono.sum(axis=-1)     : d_gt

3. GET GD style/predict encoding from ref_mels and mels
  - style_encoder(ref_mels/mels)     : ref_ss/ss(or s_dur)
  - predictor_encoder(ref_mels/mels) : ref_sp/gs
  - concate(ref_ss/ss, ref_sp/gs)    : ref/s_trg

4. GET bert-embeded text                    : For prediction
  - bert(text)                       : bert_dur
  - bert_encoder(bert_dur)           : d_en

5. [epoch>diff_epoch]  Train diffusion       (USE whole length)
  - sampler(bert_dur, ref, noise)           : s_preds
  - diffusion(s_trg, s_trg, bert_dur, ref)  : loss_diff    -> ?   
  - F.l1_loss(s_preds, s_trg)               : loss_sty

6. [epoch>0]           Train dur, pe, ce    (USE part length)
  - predictor(d_en, s_dur, s2s_attn_mono)   : d, p         # d=duration, p=aligned phoneme encoding only for dur/pe prediction?
  - cut(asr)                                : en           # prepare for training
  - cut(p)                                  : p_en
  - cut(mels)                               : gt           # for gd mels
  - cut(mels, 2len)                         : st           # for reference  (there are two reference)
  - predictor_encoder(st)                   : s_dur        # WHY NOT use gt ???
  - style_encoder(st)                       : s
  
  - pitch_extractor(gt)                     : F0_real
  - log_norm(gt)                            : N_real
  - decoder(en, F0_real, N_real, s)         : mel_rec_gt_pred   # NOT used (no grad)
  - predictor.F0Ntrain(p_en, s_dur)         : F0_fake, N_fake
  
  - smooth_l1_loss(F0_real, F0_fake)        : loss_F0_rec
  - smooth_l1_loss(N_real, N_fake)          : loss_norm_rec
  - s2s_attn_mono.sum(axis=-1)              : d_gt
  - l1(sigmoid(d), d_gt)                    : loss_dur
  - bce_with_logits(d, d_gt)                : loss_ce
  - 

7. [epoch>diff_epoch] Train discriminator (start_ds)
  - decoder(en, F0_fake, N_fake, s)         : mel_rec
  - discriminator(gt)                       : out
  - adv_loss(out, 1)                        : loss_real
  - r1_reg(out, gt)                         : loss_reg
  - discriminator(mel_rec)                  : out
  - adv_loss(out, 0)                        : loss_fake
  - loss_real + loss_fake + loss_reg        : d_loss
  - [epoch>0] criterion(mel_rec, gt)        : loss_mel

7B. [epoch>diff_epoch] Train decoder (start_ds)                 # FOR CFM
  - decoder.compute_losss(gt, en, F0_fake, N_fake, s)  : loss_cfm

8. [epoch>diff_epoch] Train generator                           # no need for CFM
  - discriminator(mel_rec)                  : out_rec, f_fake
  - adv_loss(out_rec, 1)                    : loss_adv
  - model.discriminator(gt)                 : f_real 
  - mean(abs(f_real[m][k] - f_fake[m][k]))  : loss_fm

9.  Sum all loss
  - loss_mel + loss_F0_rec+ loss_ce + loss_norm_rec + loss_dur + loss_adv + loss_fm + loss_sty + loss_diff
  - running_loss += loss_mel 

10. validate loss (mel_loss, dur_loss, F0_loss)
  - (t_en, s2s_attn_mono) -> asr;  s2s_attn_mono -> d_gt
  - mel ->pe/se: (ss, gs) -> s_trg
  - texts -> bert_dur     -> d_en
  - d_en, s ->predictor   : (d, p)
  - asr, p, mels ->cut    : (en, p_en, gt)
  - gt           ->pe     : s
  - p_en, s      ->F0train: F0_fake, N_fake   ->(F0_real)  F0_loss
  - decoder(en, F0_fake, N_fake, s): mel_rec  ->(gt)       mel_loss
  - d, d_gt                                   ->           dur_loss

11. synthesize speech(gd/pred)
  - model.decoder(en, F0_fake, N_fake, s)  : mel_pred
  - generator(mel_pred)                    : y_pred
  - writer.add_audio(y_pred)               : NONE       

12. saving model

### Train_first question
### Train_second question
- use mel_rec_gt_pred as gd before joint training in 2nd. -> why to do this?  
- gt.requires_grad_()
- add loss_cfm to g_loss, or independently?
- randomly pick whether to use in-distribution text ?
- compute the gradient norm
- ref_mels ->pe/ae(): ref[ref_ss, ref_sp] ->sampler(bert_dur) : s_preds   # sp: psd style
- mels     ->pe/ae(): s_trg[ss,     gs]     ->l1(s_trg, s_preds): loss_sty  # ss: psd style (s_dur)
- ss       ->predictor(d_en, s_dur, s2s): p 
- p->cut p_en ->predictor.F0Ntrain(p_en, ss) -> F0_fake, N_fake
- mask?


### Train_second memo
- remove adv, fm loss, d_loss loss
- Loss in 1st: cfm, mono, s2s
- Loss in 2nd: cfm, pe, dur, ce, sdiff, styF1
- updating: bert, bert_encoder, predictor, predictor_encoder, decoder
- eval, pred, gt result
- mel, ref_mel different
- 

## StyleTTS2 + mDiT (1. Flow matching, 2. LDM, 3. LFM)

### Train_first (FM)
- Load models, schedule
  - adjust bert, acoustic module lr
  - Load latest model
  - load ASR, pitch_extractor, plbert

- GET encoded text with text aligner
    - text_aligner(mel, text)        : s2s_pred, s2s_attn
    - maximum_path(s2s_attn, mask_ST): s2s_attn_mono
    - text_encoder(text):t_en
    - t_en @ s2s_attn_mono           : asr
    - s2s_attn_mono.sum(axis=-1)     : d_gt

- [1] GET GD style/predict encoding from ref_mels (part?) and mels (whole)
  - style_encoder(mels)                       : ref_ss/ss(or s_dur)
  - predictor_encoder(mels)                   : ref_sp/gs
  - concate(ref_ss/ss, ref_sp/gs)                      : ref/s_trg

- GET bert-embeded text
  - bert(text)                       : bert_dur
  - bert_encoder(bert_dur)           : d_en

- [1] GET GD pitch/energy/style
  - cut(asr)                                : en
  - cut(mels)                               : gt
  - cut(mels, 2len)                         : st
  - pitch_extractor(gt)                     : F0_real
  - style_encoder(st)                       : s       # style need longer mel ??
  - log_norm(gt)                            : N_real

- [1,epoch>0] Train mdit by FM
  - concate(F0_real, real_norm)             : r
  - decoder(mels, en, r, s)                 : vf
  - q_sample(mels, noise)                   : vf_gd
  - l_flow(vf, vf_gd)                       : loss_flow

- [epoch>0] Train dur
  - cross_entropy(s2s_pred, texts)          : loss_s2s
  - l1_loss(s2s_attn, s2s_attn_mono)        : loss_mono

- [epoch>0] Train dur, text 
  - torch.sigmoid(d)                        : _dur_pred
  - l1_loss(_dur_pred, d_gt)                : loss_dur
  - binary_cross_entropy_with_logits(d,d_gt): loss_ce

- Sum all loss
  - loss_mel + loss_flow + loss_mono + loss_s2s
  - running_loss += loss_mel 

- validate loss, synthesize speech(gd/pred)
  - (t_en, s2s_attn_mono) -> asr;  s2s_attn_mono -> d_gt
  - mel ->pe/se: (ss, gs) -> s_trg
  - texts -> bert_dur     -> d_en
  - d_en, s ->predictor   : (d, p)
  - asr, p, mels ->cut    : (en, p_en, gt)
  - gt           ->pe     : s
  - p_en, s      ->F0train: F0_fake, N_fake

  - dur_loss
  - F0_loss
  - decoder.sampler(en, F0_fake, N_fake, s)  : mel_pred
  - generator(mel_pred)                      : y_pred
  - writer.add_audio(y_pred)                 : NONE       


### Train_first (LDM)
- [2] GET GD style/predict encoding from ref_mels (part?) and mels (whole)
  - style_encoder(ref_mels/mels)                       : ref_ss/ss(or s_dur)
  - predictor_encoder(ref_mels/mels)                   : ref_sp/gs
  - concate(ref_ss/ss, ref_sp/gs)                      : ref/s_trg

- [2] GET GD pitch/energy/style
  - style_encoder(ref_mels/mels)                       : ref_ss/ss(or s_dur)
  - predictor_encoder(ref_mels/mels)                   : ref_sp/gs
  - concate(ref_ss/ss, ref_sp/gs)                      : ref/s_trg
  - [Model=LDM]first_stage_model(mels)                 : z     -> mels
  - [Model=LDM]first_stage_model(ref_mels)             : z_ref -> ref_mels
  - cut(asr)                                : en
  - cut(mels|z)                             : gt, gt_z
  - cut(mels, 2len)                         : st
  - pitch_extractor(gt)                     : F0_real
  - style_encoder(gt_z)                     : s_z
  - log_norm(gt)                            : N_real

- [2,epoch>0] Train mdit by FM
  - concate(F0_real, real_norm)             : r
  - decoder(mels, en, r, s)                 : vf
  - q_sample(mels, noise)                   : score
  - l_diff(score, noise)                       : loss_diff

- saving model


### Train_second_LDM
- Load models, schedule
  - adjust bert, acoustic module lr
  - Load latest model
  - load ASR, pitch_extractor, plbert

- GET encoded text with text aligner
    - text_aligner(mel, text):s2s_attn
    - maximum_path(s2s_attn, mask_ST): s2s_attn_mono
    - text_encoder(text):t_en
    - t_en @ s2s_attn_mono           : asr
    - s2s_attn_mono.sum(axis=-1)     : d_gt

- GET GD style/predict encoding from ref_mels (part?) and mels (whole)
  - style_encoder(ref_mels/mels)     : ref_ss/ss(or s_dur)
  - predictor_encoder(ref_mels/mels) : ref_sp/gs
  - concate(ref_ss/ss, ref_sp/gs)    : ref/s_trg

- GET bert-embeded text
  - bert(text)                       : bert_dur
  - bert_encoder(bert_dur)           : d_en

- [epoch>diff_epoch]  Train diffusion
  - sampler(bert_dur, ref, noise)           : s_preds
  - diffusion(s_trg, s_trg, bert_dur, ref)  : loss_diff    -> ?   
  - F.l1_loss(s_preds, s_trg)               : loss_sty

- [epoch>0]           Train dur/pe predictor
  - predictor(d_en, s_dur, s2s_attn_mono)   : d, p          -> ?
  - cut(asr)                                : en
  - cut(p)                                  : p_en
  - cut(mels)                               : gt
  - cut(mels, 2len)                         : st
  - predictor_encoder(st)                   : s_dur
  - style_encoder(st)                       : s
  - pitch_extractor(gt)                     : F0_real
  - log_norm(gt)                            : N_real
  - decoder(en, F0_real, N_real, s)         : mel_rec_gt_pred   # NOT used (no grad)
  - predictor.F0Ntrain(p_en, s_dur)         : F0_fake, N_fake
  - smooth_l1_loss(F0_real, F0_fake)        : loss_F0_rec
  - smooth_l1_loss(N_real, N_fake)          : loss_norm_rec

- [epoch>diff_epoch] Train LDM
  - decoder(en, F0_fake, N_fake, s)         : mel_rec
  - discriminator(gt)                       : out
  - adv_loss(out, 1)                        : loss_real
  - r1_reg(out, gt)                         : loss_reg
  - discriminator(mel_rec)                  : out
  - adv_loss(out, 0)                        : loss_fake
  - loss_real + loss_fake + loss_reg        : d_loss
  - [epoch>0] criterion(mel_rec, gt)        : loss_mel

- [epoch>0]        Train dur, asr 
  - torch.sigmoid(d)                        : _dur_pred
  - l1_loss(_dur_pred, d_gt)                : loss_dur
  - binary_cross_entropy_with_logits(d,d_gt): loss_ce

- Sum all loss
  - loss_mel + loss_F0_rec+ loss_ce + loss_norm_rec + loss_dur + loss_adv + loss_fm + loss_sty + loss_diff
  - running_loss += loss_mel 

- validate loss, synthesize speech(gd/pred)
  - mel_loss
  - dur_loss
  - F0_loss
  - model.decoder(en, F0_fake, N_fake, s)  : mel_pred
  - generator(mel_pred)                    : y_pred
  - writer.add_audio(y_pred)               : NONE       


### Variable
- z_length, z_len, z_mask
- mel_length, mel_len

### decoder
  - forward(en, r, s)                   : score
    - q_sampler(en)                     : en_q 
    - mDiT(en_q, r, s): score 
  - p_sampler(noise, en, r, s)          : mel_rec
    - SDE(nosie, en, r, s)              : mel_rec

### first_stage_model(mels)               : z
- Question
  - do wav normalizaiton before mel extraction (becuase )


### Learn later
1. scheduler_params
2. 


### Question


### Code learning


```python

# adjust BERT learning rate
parameter["modelA"]["lr"] = ? 

# build_optimizer
optimzer = build_optimizer()

# load partial ckpt
load_ckpt(model, ckpt_path, ignor_module=["module1", "module2", "module3", "module4"])

```


```python
self.F0_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))

```



        for i, batch in enumerate(train_dataloader):
            waves = batch[0]
            batch = [b.to(device) for b in batch[1:]]
            texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch

            # get z and z_lengths, ref_z and ref_z_lengths
            z, z_length = convert_mel2z(mels, model, mel_input_length)
            ref_z, ref_z_length = convert_mel2z(ref_mels, model, mel_input_length)   #### CEHCK diff between ref_mels and mels and ref_mel_input_length?


            with torch.no_grad():
                # 1. GET aligned text encoding (asr & asr_z) with text aligner
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
                asr_z = F.interpolate(asr, size=asr.size(-1) // z_rate, mode="linear", align_corners=True)
                d_gt = s2s_attn_mono.sum(axis=-1).detach()

                # 2. GET ref_z
                if multispeaker and epoch >= diff_epoch:
                    ref_ss_z = model.style_encoder(ref_z.unsqueeze(1))
                    ref_sp_z = model.predictor_encoder(ref_z.unsqueeze(1))
                    ref_z = torch.cat([ref_ss_z, ref_sp_z], dim=1)

            # 3. GET s_trg_z
            ### this operation cannot be done in batch because of the avgpool layer (may need to work on masked avgpool)
            ### Why ref_z did it in batch?
            ss, ss_z = [], []
            gs, gs_z = [], []
            for bib in range(len(z_length)):
                z = z[bib, :, :z_length[bib]]
                sz = model.predictor_encoder(z.unsqueeze(0).unsqueeze(1))
                ss_z.append(sz)
                sz = model.style_encoder(z.unsqueeze(0).unsqueeze(1))
                gs_z.append(sz)
            s_dur_z = torch.stack(ss_z).squeeze()  # global prosodic styles
            gs_z = torch.stack(gs_z).squeeze()  # global acoustic styles
            s_trg_z = torch.cat([gs_z, s_dur_z], dim=-1).detach()  # ground truth for denoiser

            # 4. GET bert-embedded text
            bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

            # Train1: Denoiser training (->z)
            if epoch >= diff_epoch:
                num_steps = np.random.randint(3, 5)
                #  model.diffusion.module.diffusion.sigma_data -> model.diffusion.diffusion.sigma_data
                if model_params.diffusion.dist.estimate_sigma_data:
                    model.diffusion.diffusion.sigma_data = s_trg_z.std(axis=-1).mean().item()  # batch-wise std estimation
                    running_std.append(model.diffusion.diffusion.sigma_data)
                if multispeaker:
                    s_preds_z = sampler(noise=torch.randn_like(s_trg_z).unsqueeze(1).to(device),
                                      embedding=bert_dur,
                                      embedding_scale=1,
                                      features=ref_z,  # reference from the same speaker as the embedding
                                      embedding_mask_proba=0.1,
                                      num_steps=num_steps).squeeze(1)
                    loss_diff = model.diffusion(s_trg_z.unsqueeze(1), embedding=bert_dur, features=ref_z).mean()  # EDM loss
                    loss_sty = F.l1_loss(s_preds_z, s_trg_z.detach())  # style reconstruction loss
                else:
                    s_preds_z = sampler(noise=torch.randn_like(s_trg_z).unsqueeze(1).to(device),
                                      embedding=bert_dur,
                                      embedding_scale=1,
                                      embedding_mask_proba=0.1,
                                      num_steps=num_steps).squeeze(1)
                    loss_diff = model.diffusion.diffusion(s_trg_z.unsqueeze(1), embedding=bert_dur).mean()  # EDM loss
                    loss_sty = F.l1_loss(s_preds_z, s_trg_z.detach())  # style reconstruction loss
            else:
                loss_sty = 0
                loss_diff = 0

            # 5. GET aligned duration
            d_z, p_z = model.predictor(d_en, s_dur_z, input_lengths, s2s_attn_mono, text_mask)  # p: aligned duration  CHECK
            mel_len = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)
            mel_len_st = int(mel_input_length.min().item() / 2 - 1)
            z_len, z_len_st = mel_len // z_rate, mel_len_st // z_rate

            with torch.no_grad():  # get
                z_length_cut = torch.where(z_length < z_len * 2, z_length, torch.tensor(z_len * 2))
                z_cut_mask = length_to_mask(z_length_cut).to('cuda')

            en, en_z = [], []
            gt, gt_z = [], []
            st, st_z = [], []
            p_en_z = []
            wav = []

            # 6. GET gt (for pe gd), gt_z (styleDiff gd/dit input), st_z (for condition styleDiff), p_en (for pe pred), and en_z (for mu)
            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)
                random_start = np.random.randint(0, mel_length - mel_len)
                random_start_z = random_start // z_rate

                en.append(asr[bib, :, random_start:random_start + mel_len])
                p_en_z.append(d_z[bib, :, random_start:random_start + z_len])
                gt.append(mels[bib, :, (random_start * 2):((random_start + mel_len) * 2)])

                en_z.append(asr_z[bib, :, random_start_z:random_start_z + z_len])
                gt_z.append(z[bib, :, (random_start_z * 2):((random_start_z + z_len) * 2)])

                y = waves[bib][(random_start * 2) * 300:((random_start + mel_len) * 2) * 300]
                wav.append(torch.from_numpy(y).to(device))

                # style reference (better to be different from the GT)
                random_start = np.random.randint(0, mel_length - mel_len_st)
                random_start_z = random_start // z_rate
                st.append(mels[bib, :, (random_start * 2):((random_start + mel_len_st) * 2)])
                st_z.append(z[bib, :, (random_start_z * 2):((random_start_z + z_len_st) * 2)])

            wav = torch.stack(wav).float().detach()
            p_en_z = torch.stack(p_en_z)
            en, en_z = torch.stack(en), torch.stack(en_z).detach()
            gt, gt_z = torch.stack(gt).detach(), torch.stack(gt_z).detach()
            st, st_z = torch.stack(st).detach(), torch.stack(st_z).detach()


            if gt.size(-1) < 80:
                continue

            # GET gd and pred of pe
            s_dur_z = model.predictor_encoder(st_z.unsqueeze(1) if multispeaker else gt_z.unsqueeze(1))  # why different on multispeaker
            s_z = model.style_encoder(st_z.unsqueeze(1) if multispeaker else gt_z.unsqueeze(1))
            with torch.no_grad():
                F0_real_z, _, F0_z = model.pitch_extractor(gt_z.unsqueeze(1))
                N_real_z = log_norm(gt_z.unsqueeze(1)).squeeze(1)
            F0_fake_z, N_fake_z = model.predictor.F0Ntrain(p_en_z, s_dur_z)

            # pe loss
            loss_F0_rec_z = (F.smooth_l1_loss(F0_real_z, F0_fake_z)) / 10
            loss_norm_rec_z = F.smooth_l1_loss(N_real_z, N_fake_z)

            # LDM loss
            cond = (en_z, F0_fake_z, N_fake_z, s_z, z_length_cut, ~z_cut_mask)
            gd_z_4d = convert_3d_to_4d(gt_z, channel=8, converted_z_dim=160)
            loss_ldm_mono_vlb, loss_dict = model.decoder(gd_z_4d, cond)  # TEMP
            loss_ldm, loss_monoMask, loss_vlb = loss_dict["train/loss_simple"], loss_dict["train/loss_monoAttn"], loss_dict["train/loss_vlb"]
