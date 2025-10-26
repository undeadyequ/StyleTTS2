## Model_vaeData_trainData_dataVersion_fineTunePhase_<psdControl>
## Model_vaeTrainData_dataVersion_fineTunePhase_<psdControl>
- Example2: dit_librittsesd_librittsesd_cutdur_phase2
  - Model:       dit
  - vaeData:     libritts (train-clean-360) + esd
  - trainData:   libritts (train-clean-360) + esd
  - dataVersion: cutdur (cut speech whose duration longer than 13s to avoid pitch/mel mismatch)
  - fineTunePhase: phase2, which is training on other ckpt (normally libritts or librittsVAE)


- Example2: dit_librittsesd_librittsesd_cutdur_phase2_infer
  - set preprocessed_data dir to esd dir 

- Example3: mdit_librittsesd_cutdurspn_pe
  - vaeTrainData: librittsesd
  - psdControl:   conditioned on pitch&energy

* Caution
  - mel is cut by [:predfined_dur]
  - phoneme pitch -> duplicated to frame-level -> downsample to latent size (by ?) -> interpolated by latent_t_size (Caution: not start from 0, which cause misalignment)

    