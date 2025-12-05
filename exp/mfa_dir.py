import argparse
import os


def mfa_dir(in_wavtxt_dir, model, out_textgrid_dir):
    dict1 = "/home/rosen/Documents/MFA/pretrained_models/dictionary/english_us_mfa.dict"
    dict2 = "~/data/lexicon/librispeech-lexicon.txt"
    os.system("/home/rosen/miniconda3/envs/aligner/bin/mfa align {} {} {} {} --clean".format(
        in_wavtxt_dir,
        dict1,
        model,
        out_textgrid_dir)
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("in_wavtxt_dir", default="~/data/test_data/")
    parser.add_argument("model", default="english_mfa")
    parser.add_argument("out_textgrid_dir", default="~/data/tg_dir/")

    args = parser.parse_args()

    mfa_dir(args.in_wavtxt_dir, args.model, args.out_textgrid_dir)