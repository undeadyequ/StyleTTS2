import os.path
from utils import recursive_munch, maximum_path, mask_from_lens, log_norm

model_n_config_dict = {
    "n1": [],
    "n2": []
}

## pe types
pe_types = ["gd_pe", "psdEnc_pe", "psdDiff_pe", "mix_pe"]
## glb types
glb_types = ["acoEnc_glb", "acoDiff_glb", "mix_glb"]
## dur types
dur_types = ["gd_dur", "psdEnc_dur", "psdDiff_dur"]


# get vocoder
vocoder = ""
def find_instable_psd_problem(model, data, pe_types=[], glb_types=[], dur_types=[]):
    pass
    # get model

    # choose cond
    for pe_type in pe_types:
        pe = get_pe(pe_type)
        for glb_type in glb_types:
            glb = get_glb(glb_type)
            for dur_type in dur_types:
                dur = get_dur(dur_type)

                output = os.path.join(exp_root_path, f"{model}_{pe_type}_{glb_type}_{dur_type}")
                # mkdir

                for i, txt in enumerate(data):
                    mel = model(txt * dur, pe, glb)
                    wav = vocoder(mel)
                    wav_f = f"{i}.wav"
                    save(wav, wav_f)


def get_pe(pe_type, model, ref_mel=None, en=None, psdEnc=None, psdDiff=None):
    """
    pe_types = ["gd_pe", "psdEnc_pe", "psdDiff_pe"]

    """
    if pe_type in ["gd_pe", "ref_pe"]:
        F0_cond, _, _ = model.pitch_extractor(ref_mel.unsqueeze(1))
        N_cond = log_norm(ref_mel.unsqueeze(1)).squeeze(1).detach()
    elif pe_type == "psdEnc_pe":
        F0_cond, N_cond = model.predictor.F0Ntrain(en, psdEnc)
    elif pe_type == "psdDiff_pe":
        F0_cond, N_cond = model.predictor.F0Ntrain(en, psdDiff)
    elif pe_type == "ref_aware_pred_pe":
        F0_ref, _, _ = model.pitch_extractor(ref_mel.unsqueeze(1))
        N_ref = log_norm(ref_mel.unsqueeze(1)).squeeze(1).detach()
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

    if pe_type == "":
        pass
    elif pe_type == "":
        pass
    elif pe_type == "":
        pass
    else:
        print("not exist")
    return ""

def get_glb(glb_type):
    if glb_type == "":
        pass
    elif glb_type == "":
        pass
    elif glb_type == "":
        pass
    else:
        print("not exist")
    return ""

def get_dur(dur_type):
    if dur_type == "":
        pass
    elif dur_type == "":
        pass
    elif dur_type == "":
        pass
    else:
        print("not exist")
    return ""


if __name__ == '__main__':
    pass
    # Test data types
    data_name_types = {"esd_unpara": ["r", "s"], "libritts_para": ["r.txt", "s.txt"]}

    # models/epoch types
    model_name, epoch = "", ""