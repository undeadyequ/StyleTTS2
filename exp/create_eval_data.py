import os
import random

def make_samples(meta_path, N, K, M, out_s="s1_v3.txt", out_r="r1_v3.txt",
                 NEED_NO_OVERLAP=True, seed=42):
    random.seed(seed)

    # Load all lines
    with open(meta_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if "|" in line]

    # Parse speaker info
    entries = []
    for line in lines:
        wav, txt = line.split("|", 1)
        spk = wav.split("/")[4]  # speaker id
        entries.append({"wav": wav, "txt": txt, "spk": spk, "line": line})

    # -------- Part 1: N random utterances (transcriptions only) --------
    samples_n = random.sample(entries, N)
    with open(out_s, "w", encoding="utf-8") as f:
        for e in samples_n:
            f.write(e["txt"] + "\n")
    print(f"Saved {N} random transcriptions to {out_s}")

    # -------- Part 2: K speakers × M utterances --------
    usable_entries = entries
    if NEED_NO_OVERLAP:
        used_ids = set(e["wav"] for e in samples_n)
        usable_entries = [e for e in entries if e["wav"] not in used_ids]
        print(f"Excluded {len(used_ids)} utterances due to NEED_NO_OVERLAP=True")

    # Group by speaker
    spk_dict = {}
    for e in usable_entries:
        spk_dict.setdefault(e["spk"], []).append(e)

    if len(spk_dict) < K:
        raise ValueError(f"Not enough speakers: have {len(spk_dict)}, need {K}")
    chosen_spks = random.sample(list(spk_dict.keys()), K)

    selected = []
    for spk in chosen_spks:
        if len(spk_dict[spk]) < M:
            print(f"Warning: Speaker {spk} has only {len(spk_dict[spk])} samples, need {M}")
            chosen = spk_dict[spk]
        else:
            chosen = random.sample(spk_dict[spk], M)
        selected.extend(chosen)

    with open(out_r, "w", encoding="utf-8") as f:
        for e in selected:
            f.write(e["line"] + "\n")
    print(f"Saved {len(selected)} (K={K}, M={M}) wav|transcriptions to {out_r}")


def make_samples_lengthratio(meta_path, N, K, M,
                 u_min=1, u_max=50,   # utterance length range (for s1_v3)
                 L_min=1, L_max=50,   # utterance length range (for r1_v3)
                 out_s="s1_v3.txt", out_r="r1_v3.txt",
                 NEED_NO_OVERLAP=True, seed=42, SYN_GENERATION=True):
    random.seed(seed)

    # Load all lines
    with open(meta_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if "|" in line]

    # Parse speaker info + length
    entries = []
    for line in lines:
        wav, txt = line.split("|", 1)
        spk = wav.split("/")[4]  # speaker id
        length = len(txt.split())  # count words
        entries.append({"wav": wav, "txt": txt, "spk": spk, "len": length, "line": line})

    # -------- Part 1: N random utterances (transcriptions only, length constrained) --------
    pool_n = [e for e in entries if u_min <= e["len"] <= u_max]
    if len(pool_n) < N:
        raise ValueError(f"Not enough utterances in [{u_min}, {u_max}]: only {len(pool_n)}, need {N}")
    samples_n = random.sample(pool_n, N)
    if SYN_GENERATION:
        with open(out_s, "w", encoding="utf-8") as f:
            for e in samples_n:
                f.write(e["txt"] + "\n")
        print(f"Saved {N} random transcriptions (len in [{u_min},{u_max}]) to {out_s}")

    # -------- Part 2: K speakers × M utterances (length constrained) --------
    usable_entries = entries
    if NEED_NO_OVERLAP:
        used_ids = set(e["wav"] for e in samples_n)
        usable_entries = [e for e in entries if e["wav"] not in used_ids]
        print(f"Excluded {len(used_ids)} utterances due to NEED_NO_OVERLAP=True")

    # Group by speaker
    spk_dict = {}
    for e in usable_entries:
        spk_dict.setdefault(e["spk"], []).append(e)

    # Find valid speakers
    valid_speakers = []
    for spk, utts in spk_dict.items():
        pool = [u for u in utts if L_min <= u["len"] <= L_max]
        if len(pool) >= M:
            valid_speakers.append(spk)

    if len(valid_speakers) < K:
        raise ValueError(f"Not enough valid speakers with >= {M} samples in [{L_min},{L_max}]")

    chosen_spks = random.sample(valid_speakers, K)

    selected = []
    for spk in chosen_spks:
        pool = [u for u in spk_dict[spk] if L_min <= u["len"] <= L_max]
        chosen = random.sample(pool, M)
        selected.extend(chosen)

    with open(out_r, "w", encoding="utf-8") as f:
        for e in selected:
            f.write(e["line"] + "\n")
    print(f"Saved {len(selected)} (K={K}, M={M}, len in [{L_min},{L_max}]) wav|transcriptions to {out_r}")


if __name__ == '__main__':
    meta_file = "/hdd/LibriTTS_16k/libri_tts_360.txt"
    if True:
        make_samples(meta_file, N=10, K=4, M=5, out_s="data2/libri_s1.txt", out_r="data2/libri_r1.txt", NEED_NO_OVERLAP=True)

    if True:
        make_samples_lengthratio(meta_file, N=2, K=4, M=10, u_min=10, u_max=12, L_min=20, L_max=24, out_s="data2/libri_s2.txt",
                                 out_r="data2/libri_r2_2.txt", NEED_NO_OVERLAP=True, SYN_GENERATION=False)