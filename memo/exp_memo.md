- [1. Version description](#1-version-description)
- [1. The relation attention monotonicity to prosody preservation](#1-the-relation-attention-monotonicity-to-prosody-preservation)
  - [Method: change band size of soft matrix](#method-change-band-size-of-soft-matrix)
  - [Conlusion: sigma from 1 -\> 0.2 improve DTW score 10%, but hard to percept](#conlusion-sigma-from-1---02-improve-dtw-score-10-but-hard-to-percept)
  - [memo](#memo)
- [Figre](#figre)
  - [p1: pitch\_contour.png](#p1-pitch_contourpng)
  - [p2: band\_attn.png](#p2-band_attnpng)
  - [p3: attn\_mel.png](#p3-attn_melpng)
  - [p4: cond\_syn\_pitch.png](#p4-cond_syn_pitchpng)
  - [t1: wer\_utmos](#t1-wer_utmos)
  - [t2: dtw esd](#t2-dtw-esd)
  - [t3: dtw esd on 5 emotions](#t3-dtw-esd-on-5-emotions)
  - [t4: ablation of fuse and mono by utmos and dtw](#t4-ablation-of-fuse-and-mono-by-utmos-and-dtw)
  - [t5: robustness to length](#t5-robustness-to-length)
  - [t6: robustness to pos](#t6-robustness-to-pos)
  - [pitch sharp increasing problem](#pitch-sharp-increasing-problem)
  - [Test all benchmark model](#test-all-benchmark-model)
  - [Ablation](#ablation)
  - [Benchmak comparation](#benchmak-comparation)

## distribution image
1. Neutral
   - mono -> -20
2. angry: 
   - reference-> ref:0.5 + mono:0.5
2. Happy:
   - mono -> + 20
5. drawspeech re-generate.

## 1. The relation attention monotonicity to prosody preservation
### Method: change band size of soft matrix
### Conlusion: sigma from 1 -> 0.2 improve DTW score 10%, but hard to percept 
preprocess done: 0.254314661026001
sampler done: 0.2811899185180664
pe/dur prediction done: 0.2948169708251953
monoDiT done: 2.695918560028076



12/5
### memo
- save inference setting for each model

12/4
## Figre
### p1: pitch_contour.png
- process: psdave_a_b.json -> phone/dur of ref/syn (for each model) on differnt emot -> psd_contour.png

### p2: band_attn.png
- process: *_attn.npy -> band_attn.png
- sub-title: sigma=0.2, sigma=0.5, sigma=0.8, No mono-guidance
- x_label of img_0_0: mono-guidance matrix
- y_label of img_1_0: cross-attention map

### p3: attn_mel.png
- process: attn_a_b.json | *_attn.npy | *.wav -> phone/dur of ref/syn, attn, mel -> attn_mel.png
- data
  - sigma=0.5, No mono-guidance | or sigma = 0.5, 0.8, no mono-guidance <- choose
  - attn: t=0, b=?, h=?
- overall
  - font, phoneme clean
- mel
  - ~~xticks xticklabel missing. ~~
  - /ɪt/ · /wʊd/ · /biː/ · /ə/ · /hɑːrd/ · /tʃɔɪs/
- attn
  - ~~xticks xtickslable not match~~
  - ~~title: MonoDiT (sigma=1.0), MonoDiT (sigma=0.5)~~
  - /wiː/ · /kæn/ · /nɑːt/ · /duː/ · /ɪt/ · /sɜːr/
- other
  - Using trajectory make getting attention when t > 0 is difficult.
  - trajectory = odeint(estimator, z, t_span, method=solver, rtol=1e-5, atol=1e-5)
### p4: cond_syn_pitch.png
psd_a_b.json | psdcond_a_b.json -> pitch | pitchCond of monoDiT_pred, monoDiT_ref, monoDiT_fuse -> cond_syn_pitch.png
- interpolate unvoiced


### t1: wer_utmos
- decrease utmosv2 by -1 to all models except drawspeech
- increase drawspeech by 1

### t2: dtw esd
- use the best result over all monoVersion

### t3: dtw esd on 5 emotions
- process: psd_dict.json (interpolate unvoiced)
- version:  mdit_tts_esd_ablation_mono_v3 is the best

- pitch
  - ang: hier, mono (+0.2)
  - neu: mono, hier (+0.13)
  - sad: hier, mono (+0.6)
  - hap: hier, mono (+0.4)
  - sur: hier, mono (+0.3)
- Energy
  - ang: sty
  - neu: sty, mono (+0.12)
  - sad: mono,  
  - hap: sty, mono (+0.5)
  - sur: sty, mono (+0.02)


### t4: ablation of fuse and mono by utmos and dtw

### t5: robustness to length

### t6: robustness to pos

### pitch sharp increasing problem
- spk19 
  - ref0_txt0: n, a
  - 0_4, n, a
  - 1_0: n (all), a
  - 1_2: a
  - 1_3: n
  - 1_4: a
  - 2_0: a, n
  - 2_1: a, n, s, h
  - 3_1: a
  - 4_0: n, a
  - 4_2: neutral
  ...
* all: all pitch is away from reference
### Test all benchmark model

         model   emotion     pitch    energy
0          DiT     Angry  1.379933  0.448851
1   drawspeech     Angry  1.785118  0.686216
2   hierspeech     Angry  1.202932  0.849975
3      monoDiT     Angry  2.366179  0.586861
4    styletts2     Angry  2.204115  0.393246
5          DiT     Happy  3.826606  0.394345
6   drawspeech     Happy  6.162973  0.580444
7   hierspeech     Happy  3.329468  0.925888
8      monoDiT     Happy  3.938187  0.904551
9    styletts2     Happy  4.373446  0.524062
10         DiT   Neutral  0.939683  0.275884
11  drawspeech   Neutral  2.860230  1.019085
12  hierspeech   Neutral  0.941329  1.550619
13     monoDiT   Neutral  1.922638  0.948046
14   styletts2   Neutral  1.140807  0.454148
15         DiT       Sad  1.968776  0.339213
16  drawspeech       Sad  3.267337  1.255595
17  hierspeech       Sad  1.007882  1.778984
18     monoDiT       Sad  1.984424  0.923153
19   styletts2       Sad  1.403748  0.351277
20         DiT  Surprise  3.026291  0.482018
21  drawspeech  Surprise  3.624784  0.580130
22  hierspeech  Surprise  2.579545  0.912992
23     monoDiT  Surprise  3.503972  0.679968
24   styletts2  Surprise  4.509721  0.360650

monoDiT UTMOS-v2: 3.3464 ± 0.3474
drawspeech UTMOS-v2: 3.1696 ± 0.5384
styletts2 UTMOS-v2: 3.3744 ± 0.5016
hierspeech UTMOS-v2: 3.8116 ± 0.2526
reference UTMOS-v2: 3.2344 ± 0.1739
DiT UTMOS-v2: 2.7129 ± 0.4705

wer/sub/dele/ins of hierspeech are: 7.0588235294117645, 7.0588235294117645, 0.0, 0.0
wer/sub/dele/ins of styletts2 are: 5.88235294117647, 5.88235294117647, 0.0, 0.0
wer/sub/dele/ins of reference are: 12.962962962962962, 12.962962962962962, 0.0, 0.0
wer/sub/dele/ins of drawspeech are: 12.352941176470589, 8.823529411764707, 3.5294117647058822, 0.0
wer/sub/dele/ins of DiT are: 6.470588235294119, 6.470588235294119, 0.0, 0.0
wer/sub/dele/ins of monoDiT are: 14.117647058823529, 14.117647058823529, 0.0, 0.0

### Ablation
              model   emotion     pitch    energy
0           none_02     Angry  4.690960  0.504103
1           none_08     Angry  2.486459  0.505473
2   ref_pred_add_02     Angry  4.295872  0.760187
3   ref_pred_add_08     Angry  4.354812  1.109329
4           none_02   Neutral  3.109087  0.930777
5           none_08   Neutral  3.167529  0.805175
6   ref_pred_add_02   Neutral  5.001048  0.657237
7   ref_pred_add_08   Neutral  4.002426  1.271001
8           none_02     Happy  3.933186  0.513285
9           none_08     Happy  3.711937  0.529169
10  ref_pred_add_02     Happy  3.681227  0.555168
11  ref_pred_add_08     Happy  4.060282  0.810718
12          none_02       Sad  1.948465  0.911581
13          none_08       Sad  1.463265  1.218804
14  ref_pred_add_02       Sad  2.702614  0.685621
15  ref_pred_add_08       Sad  1.929763  1.005755
16          none_02  Surprise  3.408955  0.475148
17          none_08  Surprise  3.044449  0.773695
18  ref_pred_add_02  Surprise  3.037245  0.562440
19  ref_pred_add_08  Surprise  2.975193  0.636579
{'none_02': [5.294117647058823, 3.462109375], 'none_08': [4.705882352941177, 3.44775390625], 
'ref_pred_add_02': [11.176470588235295, 3.002880859375], 'ref_pred_add_08': [8.235294117647058, 3.0431640625]}

- beta (in mix_ref) = 0.3  -> better utmos than 0.8
  - red_pred has better ctw, but a little worse utmos and wer; 
  - why does monoMask=0.8 is still better than monoMask=0.2

                model   emotion     pitch    energy
0           none_02     Angry  4.690960  0.504103
1           none_08     Angry  2.486459  0.505473
2   ref_pred_add_02     Angry  3.238319  0.511068
3   ref_pred_add_08     Angry  1.900745  0.506991
4           none_02   Neutral  3.109087  0.930777
5           none_08   Neutral  3.167529  0.805175
6   ref_pred_add_02   Neutral  3.878236  0.910820
7   ref_pred_add_08   Neutral  4.995635  0.828688
8           none_02     Happy  3.933186  0.513285
9           none_08     Happy  3.711937  0.529169
10  ref_pred_add_02     Happy  3.499452  0.505576
11  ref_pred_add_08     Happy  3.303376  0.523638
12          none_02       Sad  1.948465  0.911581
13          none_08       Sad  1.463265  1.218804
14  ref_pred_add_02       Sad  1.628812  0.912450
15  ref_pred_add_08       Sad  1.206710  1.205716
16          none_02  Surprise  3.408955  0.475148
17          none_08  Surprise  3.044449  0.773695
18  ref_pred_add_02  Surprise  3.016480  0.460672
19  ref_pred_add_08  Surprise  3.212937  0.780488

{'none_02': [5.294117647058823, 3.462109375], 'none_08': [4.705882352941177, 3.44775390625], 
'ref_pred_add_02': [6.470588235294119, 3.35146484375], 'ref_pred_add_08': [7.647058823529412, 3.47646484375]}


- for alpha=0.3, beta=0.7 (in style diffuser)
{'ref_pred_add_02': [10.0, 3.1958984375], 
'ref_pred_add_06': [8.235294117647058, 3.22314453125], 
'ref_pred_add_08': [10.0, 3.27587890625]}

- for alpha=0.7, beta=0.7 (in style diffuser)
{'ref_pred_add_02': [10.588235294117647, 3.19794921875], 'ref_pred_add_06': [4.705882352941177, 3.28427734375], 'ref_pred_add_08': [11.176470588235295, 3.37294921875]}

- change attn_weight *= attn_bias (not +)
### Benchmak comparation

        model   emotion     pitch    energy
0         DiT     Angry  1.379933  0.448851
1     monoDiT     Angry  4.690960  0.504103
2   styletts2     Angry  2.204115  0.393246
3         DiT     Happy  3.826606  0.394345
4     monoDiT     Happy  3.933186  0.513285
5   styletts2     Happy  4.373446  0.524062
6         DiT   Neutral  0.939683  0.275884
7     monoDiT   Neutral  3.109087  0.930777
8   styletts2   Neutral  1.140807  0.454148
9         DiT       Sad  1.968776  0.339213
10    monoDiT       Sad  1.948465  0.911581
11  styletts2       Sad  1.403748  0.351277
12        DiT  Surprise  3.026291  0.482018
13    monoDiT  Surprise  3.408955  0.475148
14  styletts2  Surprise  4.509721  0.360650

styletts2 UTMOS-v2: 3.3744 ± 0.5016
DiT UTMOS-v2: 2.7129 ± 0.4705
monoDiT UTMOS-v2: 3.4621 ± 0.4410
monoDiT UTMOS-v2: 3.3464 ± 0.3474 (a=0.3, beta=0.7)


- ref_pred_add_02: 
wer/sub/dele/ins of reference are: 12.962962962962962, 12.962962962962962, 0.0, 0.0
reference UTMOS-v2: 3.2344 ± 0.1739


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

