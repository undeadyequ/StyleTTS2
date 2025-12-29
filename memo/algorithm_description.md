## General Alogrithm
1. Text_aligner: s2s_attn = model.text_aligner(mels, mask, texts)
    - text_aligner downsampled mels to half (for better performance). so s2s_attn has half mel length (asr or en also half mels)
    ```c
    self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                  kernel_size=kernel_size, stride=stride,
                                  padding=padding, dilation=dilation,
                                  bias=bias)
    ```
2. PE prediction: F0_fake, N_fake = predictor.F0Ntrain(en, s_dur)
   - en has half mel len, while F0Ntrain unsample it to mel length

3. Style Encoding:  ref_ss, ref_sp = style_encoder(ref_mels), predictor_encoder(ref_mels)
4. Duration prediction: 
5. Text embedding:


### Style sampler
- **Target**: Stablize acoustic and prosodic embedding conditioned on text
- **Process**: noise ->(sin_emb) ->add(spk?) ->duplicate ->concat(bert_dur) ->transformer(ref) ->adapt_ave_pool
- Hyper
  - embedding_scale:  classifier-free guidance scale, The higher the scale, the more conditional the style.
  - embedding_mask_proba: ?
  - num_tep:  the higher the stpes, the more diverse the samples are

```python
s_pred = sampler(noise, embedding=bert_dur, embedding_scale=1, embedding_mask_proba=0.1, features=ref, num_step=rand(3, 5))
```


### Loss
1. EDM loss: loss_diff = model.diffusion(s_trg.unsqueeze(1), embedding=bert_dur, features=ref).mean()
    - Denoiser K(s; t,sigma) function as sigma scale normalization (noise scale is different) and transformer for EDM loss
    - EDM loss $L_{edm} = E_{x,t,\sigma, \epsilon~N(0, 1)}(\gamma(\sigma) * (K - E(x)))$
    - EDM loss is euclidean to gt, not noise
2. Duration loss


## Variable
1. The parameter declaration (first) 
   - asr: mu (aligned phoneme )     -> with half len of mel. (t_en: txt embedding)
   - en : sliced asr
   - gt : sliced mel                -> extracting pe, style_encoder input (single Speaker), training target
   - st : sliced mel (diff start)   -> style_encoder input (multi speaker)   # gt and st have different random_start
   - z  : first_stage processed mel ->
2. The parameter declaration (second)
   - asr   : 
   - ref_ss: 
   - ref_sp: 
   - ref   : 
