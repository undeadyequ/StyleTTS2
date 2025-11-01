import torch
import utmosv2
import random
import numpy as np

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# Force deterministic CuDNN ops
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------- Load model ----------
model = utmosv2.create_model(pretrained=True)
model.eval()  # disable dropout, batchnorm randomness


model = utmosv2.create_model(pretrained=True)

#mos = model.predict(input_path="/path/to/wav/file.wav")

mos_list = model.predict(input_dir="/home/rosen/Project/StyleTTS2/res")

scores = [item["predicted_mos"] for item in mos_list]
mean_mos = sum(scores) / len(scores)


# 3.00662890625 for styleTTS2
print(f"Average UTMOS-v2 score: {mean_mos:.4f}")

#
# Average UTMOS-v2: 2.9714 ± 0.4110
# Files evaluated: 250
