## Basic logic of ICL version
### 1. Train
   1. mel * (1 - rand_span_mask) -> cond
   2. torch.cat((x, cond, text_embed), dim=1) -> x
   3. loss * rand_span_mask -> masked_loss

### 2. Inference
   1. concate(txt_frm_ref, txt_frm_tgt) -> txt_frm_full    
      1. style_encoder(txt_ref) -> ref_emb * ref_s2s -> txt_frm_ref
      2. style_encoder(txt_tgt) -> tgt_emb * tgt_dur -> txt_frm_tgt
   2. concate(mel_ref, mel_mask_tgt) -> mel_full
   4. concate(pe_ref, pe_ref) -> pe_full
   5. estimator(cond=mel_full, mu=txt_frm_full, seq_style=pe_full)  [B, C, T], [B, D, T/2], [B, 2, T]
      1. noise(mel_full) -> noise_full  # len(mel_ref), len(txt_frm_tgt) -> noise_ref/tgt


### 3. Inference (during training)
   1. split tgt_emb * tgt_s2s to 30% for ref and 70% for tgt, or 30% for ref and 100% for tgt (after epoch15)

## Process details of ICL version
### 1. InferenceAPI_bertFusion_icl.py
    [ICL sliced mel/text]
    - Tokenise (text, ref_text): ps, ps_ref
    - Clip reference mel (ref): **ref_mel_sliced**
    - Force-align (ref_text, ref_mel): s2s_attn_mono_ref
    - Frame-level reference text encoding&Clip (s2s_attn_mono_ref): **en_ref**  # may mismatch with ref_mel_sliced due to the FA error
    [Frame-level Embedding]
    - Semantic/Acoustic text encode:  t_en, d_en ->(s) d
    - Prosody prediction style (ref): s
    - Duration prediction (d_en): pred_aln_trg
    - Tgt frame embedding (t_en, pred_aln_trg): **mu_tgt**
    [PE prediction]
    - Voiced mask (ps, ps_ref)  ->  uv, uv_ref
    - PE extract (ref, ref_slice): pe_extractor(ref_mel) -> F0_ref, pe_extractor(ref_mel_sliced) -> pe_ref
    - Semantic-Prosodic style ([d * pred_aln_trg], [F0_ref, s2s_ref, uv_ref, uv, pred_aln_trg]): sem_pros_style
    - PE prediction (sem_pros_style, s): **pe_tgt**
    - decoder(mu_tgt=mu_tgt, cond_ref=ref_mel_sliced, mu_ref=en_ref, seq_style_tgt=pe_tgt, seq_style_ref=pe_ref)
2. Others in InferenceAPI_bertFusion_icl.py
   - No txt_frm Shift

## Prompt to create icl (in-context learning) version one by one
1. General context (DeCoDiT-TTS and F5-TTS)

There are two Dit-based Flow matching TTS models. 
One is the proposed DeCoDiT-TTS, which is mainly implemented in @flow_mathcing_v3.py, based on the cross_DiT. The input/output of compute_loss function are
INPUT: melspectrogram (x1), frame-level text embedding (mu) and pitch_energy_contours (seq_style), and global style (c)
OUTPUT: loss of velocity field of whole melspectrogram
Process:
- concatenate(noised(x1), mu) -> x; 
- CrossDiT(x, seq_style, c, t) -> v; 
- v - u -> loss

The another is the F5-TTS, whic his mainly implemented in cfg.py in http. The input/output of forward function are:
- INPUT: melspectrogram (inp), text
- OUTPUT: loss of velocity field of whole melspectrogram
- Process:
  - concatenate(x, cond, text_embed) -> x; conv_pos_embed(x) + x -> x, where x is noised(inp). cond is masked inp 
  - DiT(x, t) -> v; 
  - mask(v - u) -> loss

2. flow_matching_v4.py (compute_loss)

I want to introduce the masked mel-spectrogram mechanism of F5-TTS to DeCoDiT-TTS, so the new input/output of compute_loss function will be
- INPUT: melspectrogram (x1), frame-level text embedding (mu) and pitch_energy_contours (seq_style) -> c is not needed anymore
- OUTPUT: loss of velocity field of whole melspectrogram
- Process: 
  - concatenate(noised(x1), cond, mu) -> x; conv_pos_embed(x) + x -> x (I don't know if it is needed?)
  - CrossDiT(x, seq_style, t) -> v;
  - mask(v - u) -> loss
Write flow_matching_v4.py and the corresponding script, such as estimator_v4.py to implement this.

3. flow_matching_v4.py (inference)

Let's consider the input/output in inference (forward function) in @ carefully. The current INPUT/OUTPUT and process are
- INPUT: frame-level target text embedding (mu), frame-level target pitch/energy contours (seq_style), reference mel frames (cond)
- OUTPUT: output_mel, attn_maps
- Process: 
  - concatenate(noise, cond, mu) -> x
  - CrossDiT(x, seq_style, t) -> v
  - ODE(v) -> output_mel

The problems are
- INPUT: mu_tgt, seq_style_tgt, cond_ref -> mu_tgt, mu_ref, seq_style_tgt, seq_style_ref, cond_ref (Comment: not only target mu, seq_style, but also reference) 
- Process
  - it should be concatenate(noise_ref_tgt, cond_ref_tgt, mu_ref_tgt) -> x, where noise_ref_tgt are concatenation of noise_ref and noise_tgt on temporal dim, same as cond, and mu. (cond_tgt is just mask) 
  - check the cfm.py and dit.py for details.

4. train_first_txt2mel_icl.py 
  Now, let's write a new train_first_txt2mel_cfm_icl.py, based on @train_first_txt2mel_cfm.py by modifying 
  - training by flow_matching_v4.py, chosen by a new config_libritts_txt2mel_cfm_v35.yml 
  - during inference, the utterance is split into first half (target) and second half (reference) by mel frame count. 
  If you have question, just ask

5. train_second_txt2mel_cfm_bertfusion_icl.py

6. inferenceAPI_bertFusion
Let's create a new inferenceAPI_bertFusion_icl.py, based on @inferenceAPI_bertFusion.py. Require
- the logic is similar as the inference logic in @train_second...py
- the mu_ref is the frame-extended ref_txt encoding given phoneme2frame duration which is provided by the force alignment between ref_mel and ref_text, refering to the "s2s_attn_mono" in train_second
- Slice ref_mel to a pre-defined length (if it is longer than that), and use give the sliced ref_mel to decoder
- Slice ref_txt adaptively to the sliced ref_mel by using the phoneme2frame duration

If you have question, just ask
