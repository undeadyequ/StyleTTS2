├── data                                          ->  style/text for synthesis 
│   ├── eval50_paratxt.txt
├── result50                                      ->  50=textNumber
│   ├── mdit_tts_esd
│   │   ├── att_A_B_C.json                   # Attention visualization {"spk": {"emo": {"A/B": {"ids/qkdurs/qkphones":... }}}}
│   │   ├── psd_A_B_C.json                   # original pit/eng/dur contours {"spk": {"emo": {"A/B/R": {"ids/dur/phones/psd":...}}}} 
│   │   ├── dtwIntp_A_B_C.json               # dtw of each contour {"spk": {"emo": {"A/B": [[pdiff_w1, ...], [ediff_w1, ...], [w1, ...]]}}}
│   │   ├── stats_A_B_C.json                 # stats over each model {"spk": {"emo": {"A/B": [p1, e1, m1]}}}   <- pitch, energy, mcd
│   │   ├── stats_A_B_C_mean.json            # stats over each model/spk {"emo": {"A/B": [p1, e1, m1]}}   <- pitch, energy, mcd
│   │   ├── vis_pitch_A_B_C.json             # pit/eng contour vis  {"emo": {"A/B": [[pit1, ..], [p1, ..]]}}
│   │   ├── wer_utmos.json                   # WER and UTMOS result {"A": [wer, sub, del, inser, utmosv2_mean, utmosv2_std]}

- variance
│   │   ├── psdave_A_B_C.json
│   │   ├── statsave_A_B_C.json                   # calucate based on phoneme-level pitch and enery
│   │   ├── statsIntp_A_B_C.csv                   # calucate only on voiced part
│   │   ├── statsIntp_A_B_C_mean.csv              # take average on spk
│   │   ├── statsIntp_psd_mulitindex_cmpM.csv     # modified champion data by pivot (multiindex) over emotion for each model
│   │   ├── statsIntp_psd_mulitindex_cmpM_meanModel.csv     # 
│   │   ├── vis_pitch_A_B_C.csv
│   │   ├── vis_energy_A_B_C.csv            




datastructure

