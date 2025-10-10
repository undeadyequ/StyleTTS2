
ipex flag is deprecated, will be removed in Accelerate v1.10. From 2.7.0, PyTorch has all needed optimizations for Intel CPU and XPU.
/home/rosen/anaconda3/envs/styletts2/lib/python3.9/site-packages/librosa/util/files.py:10: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  from pkg_resources import resource_filename
bert loaded
bert_encoder loaded
predictor loaded
decoder loaded
text_encoder loaded
predictor_encoder loaded
style_encoder loaded
diffusion loaded
text_aligner loaded
pitch_extractor loaded
mpd loaded
msd loaded
wd loaded
BERT AdamW (
Parameter Group 0
    amsgrad: False
    base_momentum: 0.85
    betas: (0.9, 0.99)
    capturable: False
    decoupled_weight_decay: True
    differentiable: False
    eps: 1e-09
    foreach: None
    fused: None
    initial_lr: 1e-05
    lr: 1e-05
    max_lr: 2e-05
    max_momentum: 0.95
    maximize: False
    min_lr: 0
    weight_decay: 0.01
)
decoder AdamW (
Parameter Group 0
    amsgrad: False
    base_momentum: 0.85
    betas: (0.0, 0.99)
    capturable: False
    decoupled_weight_decay: True
    differentiable: False
    eps: 1e-09
    foreach: None
    fused: None
    initial_lr: 1e-05
    lr: 1e-05
    max_lr: 2e-05
    max_momentum: 0.95
    maximize: False
    min_lr: 0
    weight_decay: 0.0001
)
Epoch [1/30], Step [10/70717], Loss: 0.23169, Disc Loss: 0.00000, Dur Loss: 0.48937, CE Loss: 0.02132, Norm Loss: 0.36729, F0 Loss: 2.16594, LM Loss: 0.97703, Gen Loss: 0.00000, Sty Loss: 0.00000, Diff Loss: 0.00000, DiscLM Loss: 0.00000, GenLM Loss: 0.00000
Time elasped: 7.642729997634888
Epoch [1/30], Step [20/70717], Loss: 0.24268, Disc Loss: 0.00000, Dur Loss: 0.46327, CE Loss: 0.01920, Norm Loss: 0.31681, F0 Loss: 1.45344, LM Loss: 0.79474, Gen Loss: 0.00000, Sty Loss: 0.00000, Diff Loss: 0.00000, DiscLM Loss: 0.00000, GenLM Loss: 0.00000
Time elasped: 14.032981157302856
Epoch [1/30], Step [30/70717], Loss: 0.22794, Disc Loss: 0.00000, Dur Loss: 0.36707, CE Loss: 0.01626, Norm Loss: 0.45249, F0 Loss: 1.62999, LM Loss: 0.91601, Gen Loss: 0.00000, Sty Loss: 0.00000, Diff Loss: 0.00000, DiscLM Loss: 0.00000, GenLM Loss: 0.00000
Time elasped: 20.121304035186768
Epoch [1/30], Step [40/70717], Loss: 0.23830, Disc Loss: 0.00000, Dur Loss: 0.42907, CE Loss: 0.01993, Norm Loss: 0.31095, F0 Loss: 1.48480, LM Loss: 0.79936, Gen Loss: 0.00000, Sty Loss: 0.00000, Diff Loss: 0.00000, DiscLM Loss: 0.00000, GenLM Loss: 0.00000
Time elasped: 26.110721826553345


Time elasped: 19730.76074528694
Epoch [1/50], Step [35320/35358], Mel Loss: 0.43358, Gen Loss: 0.00000, Disc Loss: 0.00000, Mono Loss: 0.00000, S2S Loss: 0.00000, SLM Loss: 0.00000
Time elasped: 19736.25603914261
Epoch [1/50], Step [35330/35358], Mel Loss: 0.43102, Gen Loss: 0.00000, Disc Loss: 0.00000, Mono Loss: 0.00000, S2S Loss: 0.00000, SLM Loss: 0.00000
Time elasped: 19741.46880054474
Epoch [1/50], Step [35340/35358], Mel Loss: 0.43263, Gen Loss: 0.00000, Disc Loss: 0.00000, Mono Loss: 0.00000, S2S Loss: 0.00000, SLM Loss: 0.00000
Time elasped: 19746.676684856415
Epoch [1/50], Step [35350/35358], Mel Loss: 0.42715, Gen Loss: 0.00000, Disc Loss: 0.00000, Mono Loss: 0.00000, S2S Loss: 0.00000, SLM Loss: 0.00000
Time elasped: 19752.148567438126
Traceback (most recent call last):
  File "/home/rosen/Project/StyleTTS2/train_first.py", line 445, in <module>
    main()
  File "/home/rosen/anaconda3/envs/styletts2/lib/python3.9/site-packages/click/core.py", line 1161, in __call__
    return self.main(*args, **kwargs)
  File "/home/rosen/anaconda3/envs/styletts2/lib/python3.9/site-packages/click/core.py", line 1082, in main
    rv = self.invoke(ctx)
  File "/home/rosen/anaconda3/envs/styletts2/lib/python3.9/site-packages/click/core.py", line 1443, in invoke
    return ctx.invoke(self.callback, **ctx.params)
  File "/home/rosen/anaconda3/envs/styletts2/lib/python3.9/site-packages/click/core.py", line 788, in invoke
    return __callback(*args, **kwargs)
  File "/home/rosen/Project/StyleTTS2/train_first.py", line 381, in main
    y_rec = model.decoder(en, F0_real, real_norm, s)
  File "/home/rosen/anaconda3/envs/styletts2/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/rosen/anaconda3/envs/styletts2/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/rosen/anaconda3/envs/styletts2/lib/python3.9/site-packages/accelerate/utils/operations.py", line 818, in forward
    return model_forward(*args, **kwargs)
  File "/home/rosen/anaconda3/envs/styletts2/lib/python3.9/site-packages/accelerate/utils/operations.py", line 806, in __call__
    return convert_to_fp32(self.model_forward(*args, **kwargs))
  File "/home/rosen/anaconda3/envs/styletts2/lib/python3.9/site-packages/torch/amp/autocast_mode.py", line 44, in decorate_autocast
    return func(*args, **kwargs)
  File "/home/rosen/Project/StyleTTS2/Modules/hifigan.py", line 458, in forward
    F0 = self.F0_conv(F0_curve.unsqueeze(1))
  File "/home/rosen/anaconda3/envs/styletts2/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/rosen/anaconda3/envs/styletts2/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1857, in _call_impl
    return inner()
  File "/home/rosen/anaconda3/envs/styletts2/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1805, in inner
    result = forward_call(*args, **kwargs)
  File "/home/rosen/anaconda3/envs/styletts2/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 375, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/home/rosen/anaconda3/envs/styletts2/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 370, in _conv_forward
    return F.conv1d(
RuntimeError: Given groups=1, weight of size [1, 1, 3], expected input[1, 300, 1] to have 1 channels, but got 300 channels instead
Traceback (most recent call last):
  File "/home/rosen/anaconda3/envs/styletts2/bin/accelerate", line 8, in <module>
    sys.exit(main())
  File "/home/rosen/anaconda3/envs/styletts2/lib/python3.9/site-packages/accelerate/commands/accelerate_cli.py", line 50, in main
    args.func(args)
  File "/home/rosen/anaconda3/envs/styletts2/lib/python3.9/site-packages/accelerate/commands/launch.py", line 1199, in launch_command
    simple_launcher(args)
  File "/home/rosen/anaconda3/envs/styletts2/lib/python3.9/site-packages/accelerate/commands/launch.py", line 785, in simple_launcher
    raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)
subprocess.CalledProcessError: Command '['/home/rosen/anaconda3/envs/styletts2/bin/python3.9', 'train_first.py', '--config_path', './Configs/config_libritts.yml']' returned non-zero exit status 1.