v1: no cfm_loss at start
v2: double grad                        -> wierd prosody
v3: no double grad  (train on 3090)    
v4: style dimension 256 -> 128         -> current best but y/pred not good
v5: style dimension 256 + theta_data 0.2 to 0.15 (train on 3090)  -> no better than 128
v6: upsample mel v2 (128)              -> training
v8: no diff                            -> y/pred bad but inference good, sampler still performance the best
v9: frozen styleEnc


v9_14: 
s_trg_norm, s_preds_norm, loss_sty:  12.412075996398926 10.648037910461426 0.06222817301750183
x_denoised, x, losses, sigmas 12.40121841430664 12.44285774230957 0.382843941450119 tensor([0.0443, 0.0349, 0.1944, 0.0587, 0.1901, 0.0643, 0.1491, 0.0150, 0.0123,
        0.0550, 0.0511, 0.2055, 0.0144, 0.0330, 0.0383, 0.2798],
       device='cuda:0')

v9_28
s_trg_norm, s_preds_norm, loss_sty:  10.613546371459961 8.964037895202637 0.05933920294046402
x_denoised, x, losses, sigmas 10.559554100036621 10.781659126281738 0.22023238241672516 tensor([0.0602, 0.2324, 0.0719, 0.0033, 0.0100, 0.0121, 0.0655, 0.0360, 0.0086,
        0.0254, 0.0346, 0.4340, 0.1500, 0.0899, 0.0942, 0.0025],
       device='cuda:0')


gan_epoch14
x_denoised, x, losses, sigmas 
13.087894439697266 13.251322746276855 0.2857, 
[0.0400, 0.3766, 0.0026, 0.0373, 0.0564, 0.1852, 0.0986, 0.0906, 0.1337, 0.0339, 0.0471, 0.0185]



gan_epoch_28
- s_trg_norm, s_preds_norm, loss_sty:  9.6311, 7.8149 0.0793
- x_denoised, x, losses, sigmas 
  - 9.081239700317383 9.631073951721191 0.3362 tensor([0.0115, 0.0435, 0.0575, 0.2923, 0.1206, 0.2148, 0.0858, 0.0862, 0.0435,
          0.1769, 0.0964, 0.0860], device='cuda:0')

