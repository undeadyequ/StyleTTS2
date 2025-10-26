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
