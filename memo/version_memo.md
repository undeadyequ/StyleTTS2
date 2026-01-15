## 1. Version description

v1: no cfm_loss at start
v2: double grad                        -> wierd prosody
v3: no double grad  (train on 3090)    
v4: style dimension 256 -> 128         -> current best but y/pred not good
v5: style dimension 256 + theta_data 0.2 to 0.15 (train on 3090)  -> no better than 128
v6: upsample mel v2 (128)              -> training
v8: no diff                            -> y/pred bad but inference good, sampler still performance the best
v9: frozen styleEnc
v10: cmp

12/24

### residual prosody fusion
pi_ref = softcap(e(t) * a(t), 0.3)
e(t) = sigmoid(text, style)
a(t) = sigmoid(diff, l2)

s2. pi_ref = soft-cap(sigmoid(e(t) * a(t)) / s), where a(t) = ga(concate(diff, l2)), e(t) = ge(text, si) 
s3. pi_ref = soft-cap(sigmoid(l2 - m) / s)
