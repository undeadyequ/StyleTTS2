1. ASRCNN compress mels to half to improve better performance. so asr (en) has half mel length
    self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                kernel_size=kernel_size, stride=stride,
                                padding=padding, dilation=dilation,
                                bias=bias)

2. The parameter declaration (first) 
   - asr: mu (aligned phoneme )     -> with half len of mel. (t_en: txt embedding)
   - en : sliced asr
   - gt : sliced mel                -> extracting pe, style_encoder input (single Speaker), training target
   - st : sliced mel (diff start)   -> style_encoder input (multi speaker)   # gt and st have different random_start
   - z  : first_stage processed mel ->

3. The parameter declaration (second)
   - asr   : 
   - ref_ss: 
   - ref_sp: 
   - ref   :

4. Ground truth from reconstruction
   - sdf


10/27
1. F0, Energy prediction values
   - F0_model = JDCNet(num_class=1, seq_len=192)
2. PE conditioning
   - Text + S(styleDiff)
   - Reference PE directly
3. Inference code
   - Text, Ref, 
4. Other
   - mask problem

10/28
1. Test ref_pe and pred_pe conditioning, expect that pred_pe is good to show monotonic?
   - if Not: considering adat_pe
2. Test stylediff and styleEnc conditioning, to decide which one to use.
3. Add synthesize speech code in main

mdit_cfm_v2_stylediff_pred_pe_epoch28_completeStyleDiff: 2.4 + 0.6