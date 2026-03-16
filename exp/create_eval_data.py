import os
import re
import random
import shutil

def make_samples(meta_path, N, K, M, out_s="s1_v3.txt", out_r="r1_v3.txt",
                 NEED_NO_OVERLAP=True, seed=42, txt_min=None):
    """
    N sentence * K speaker * M emotions
    txt_min : optional minimum word count for both synthesis texts and reference utterances
    """
    random.seed(seed)

    # Load all lines
    with open(meta_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if "|" in line]

    # Parse speaker info
    entries = []
    for line in lines:
        wav, txt = line.split("|", 1)
        spk = wav.split("/")[-3]  # speaker id
        length = len(txt.split())
        entries.append({"wav": wav, "txt": txt, "spk": spk, "len": length, "line": line})

    # -------- Part 1: N random utterances (transcriptions only) --------
    pool_n = [e for e in entries if e["len"] >= txt_min] if txt_min is not None else entries
    if len(pool_n) < N:
        raise ValueError(f"Not enough utterances with >= {txt_min} words: only {len(pool_n)}, need {N}")
    samples_n = random.sample(pool_n, N)
    with open(out_s, "w", encoding="utf-8") as f:
        for e in samples_n:
            f.write(e["txt"] + "\n")
    print(f"Saved {N} random transcriptions to {out_s}")

    # -------- Part 2: K speakers × M utterances --------
    usable_entries = [e for e in entries if e["len"] >= txt_min] if txt_min is not None else entries
    if NEED_NO_OVERLAP:
        used_ids = set(e["wav"] for e in samples_n)
        usable_entries = [e for e in usable_entries if e["wav"] not in used_ids]
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
                             tgt_ranges,
                             ref_txt_min=1, ref_txt_max=50,
                             out_s_pattern="data2/fine_libri_syn_{label}.txt",
                             out_r="data2/fine_libri_ref.txt",
                             NEED_NO_OVERLAP=True, seed=42):
    """
    LibriTTS fine test: one syn file per entry in tgt_ranges, one shared ref file.
    tgt_ranges : list of (label, tgt_min, tgt_max)
                 e.g. [("ratio05", 4, 8), ("ratio1", 8, 12), ("ratio2", 14, 20)]
    ref_txt_min/max : word-count range for the single reference file
    out_s_pattern   : filename pattern; {label} is replaced per entry
    """
    random.seed(seed)

    with open(meta_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if "|" in line]

    entries = []
    for line in lines:
        wav, txt = line.split("|", 1)
        spk = wav.split("/")[-3]
        length = len(txt.split())
        entries.append({"wav": wav, "txt": txt, "spk": spk, "len": length, "line": line})

    # -------- Part 1: one syn file per tgt_range --------
    all_used_wavs = set()
    for label, tgt_min, tgt_max in tgt_ranges:
        pool_n = [e for e in entries if tgt_min <= e["len"] <= tgt_max]
        if len(pool_n) < N:
            raise ValueError(f"[{label}] Not enough utterances in [{tgt_min},{tgt_max}]: only {len(pool_n)}, need {N}")
        samples_n = random.sample(pool_n, N)
        out_s = out_s_pattern.format(label=label)
        with open(out_s, "w", encoding="utf-8") as f:
            for e in samples_n:
                f.write(e["txt"] + "\n")
        print(f"Saved {N} transcriptions (len in [{tgt_min},{tgt_max}]) to {out_s}")
        all_used_wavs.update(e["wav"] for e in samples_n)

    # -------- Part 2: single shared ref file --------
    usable_entries = entries
    if NEED_NO_OVERLAP:
        usable_entries = [e for e in entries if e["wav"] not in all_used_wavs]
        print(f"Excluded {len(all_used_wavs)} utterances due to NEED_NO_OVERLAP=True")

    spk_dict = {}
    for e in usable_entries:
        spk_dict.setdefault(e["spk"], []).append(e)

    valid_speakers = [
        spk for spk, utts in spk_dict.items()
        if len([u for u in utts if ref_txt_min <= u["len"] <= ref_txt_max]) >= M
    ]
    if len(valid_speakers) < K:
        raise ValueError(f"Not enough valid speakers with >= {M} samples in [{ref_txt_min},{ref_txt_max}]")

    chosen_spks = random.sample(valid_speakers, K)
    selected = []
    for spk in chosen_spks:
        pool = [u for u in spk_dict[spk] if ref_txt_min <= u["len"] <= ref_txt_max]
        selected.extend(random.sample(pool, M))

    with open(out_r, "w", encoding="utf-8") as f:
        for e in selected:
            f.write(e["line"] + "\n")
    print(f"Saved {len(selected)} (K={K}, M={M}, len in [{ref_txt_min},{ref_txt_max}]) wav|transcriptions to {out_r}")


def make_samples_esd(meta_path, N, K, E, M,
                     out_s="random_s_esd.txt", out_r="random_r_esd.txt",
                     NEED_NO_OVERLAP=True, seed=42):
    """
    ESD random test.
    N  : number of synthesis utterances (text only)
    K  : number of speakers
    E  : number of emotions per speaker
    M  : number of reference utterances per (speaker, emotion)
    ref file has K*E*M lines: wav|txt
    """
    random.seed(seed)

    with open(meta_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if "|" in line]

    entries = []
    for line in lines:
        wav, txt = line.split("|", 1)
        spk = wav.split("/")[-3]
        emo = wav.split("/")[-2]
        entries.append({"wav": wav, "txt": txt, "spk": spk, "emo": emo, "line": line})

    # Part 1: N random synthesis texts
    # Deduplicate by text first: ESD records each sentence across multiple speakers/emotions.
    unique_pool = list({e["txt"]: e for e in entries}.values())
    if len(unique_pool) < N:
        raise ValueError(f"Not enough unique texts: only {len(unique_pool)}, need {N}")
    samples_n = random.sample(unique_pool, N)
    with open(out_s, "w", encoding="utf-8") as f:
        for e in samples_n:
            f.write(e["txt"] + "\n")
    print(f"Saved {N} unique transcriptions to {out_s}")

    # Part 2: K speakers × E emotions × M utterances
    usable_entries = entries
    if NEED_NO_OVERLAP:
        used_ids = set(e["wav"] for e in samples_n)
        usable_entries = [e for e in entries if e["wav"] not in used_ids]
        print(f"Excluded {len(used_ids)} utterances due to NEED_NO_OVERLAP=True")

    # Group by (speaker, emotion)
    spk_emo_dict = {}
    for e in usable_entries:
        spk_emo_dict.setdefault(e["spk"], {}).setdefault(e["emo"], []).append(e)

    # Find speakers that have at least E emotions each with >= M samples
    valid_speakers = []
    for spk, emo_dict in spk_emo_dict.items():
        valid_emos = [emo for emo, utts in emo_dict.items() if len(utts) >= M]
        if len(valid_emos) >= E:
            valid_speakers.append(spk)

    if len(valid_speakers) < K:
        raise ValueError(f"Not enough valid speakers: have {len(valid_speakers)}, need {K}")

    chosen_spks = random.sample(valid_speakers, K)

    selected = []
    for spk in chosen_spks:
        emo_dict = spk_emo_dict[spk]
        valid_emos = [emo for emo, utts in emo_dict.items() if len(utts) >= M]
        chosen_emos = random.sample(valid_emos, E)
        for emo in chosen_emos:
            chosen = random.sample(emo_dict[emo], M)
            selected.extend(chosen)

    with open(out_r, "w", encoding="utf-8") as f:
        for e in selected:
            f.write(e["line"] + "\n")
    print(f"Saved {len(selected)} (K={K}, E={E}, M={M}) wav|transcriptions to {out_r}")


def make_samples_esd_lengthratio(meta_path, N, K, E, M,
                                  tgt_ranges,
                                  ref_txt_min=1, ref_txt_max=50,
                                  out_s_pattern="data2/fine_esd_syn_{label}.txt",
                                  out_r="data2/fine_esd_ref.txt",
                                  NEED_NO_OVERLAP=True, seed=42):
    """
    ESD fine test: one syn file per entry in tgt_ranges, one shared ref file.
    N  : number of synthesis utterances per label
    K  : number of speakers
    E  : number of emotions per speaker
    M  : number of reference utterances per (speaker, emotion)
    tgt_ranges      : list of (label, tgt_min, tgt_max)
                      e.g. [("ratio05", 4, 8), ("ratio1", 8, 12), ("ratio2", 14, 20)]
    ref_txt_min/max : word-count range for the single reference file
    out_s_pattern   : filename pattern; {label} is replaced per entry
    """
    random.seed(seed)

    with open(meta_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if "|" in line]

    entries = []
    for line in lines:
        wav, txt = line.split("|", 1)
        spk = wav.split("/")[-3]
        emo = wav.split("/")[-2]
        length = len(txt.split())
        entries.append({"wav": wav, "txt": txt, "spk": spk, "emo": emo, "len": length, "line": line})

    # Part 1: one syn file per tgt_range
    # Deduplicate by text first: ESD records each sentence across multiple speakers/emotions.
    all_used_wavs = set()
    for label, tgt_min, tgt_max in tgt_ranges:
        pool_n = list({e["txt"]: e for e in entries if tgt_min <= e["len"] <= tgt_max}.values())
        if len(pool_n) < N:
            raise ValueError(f"[{label}] Not enough unique texts in [{tgt_min},{tgt_max}]: only {len(pool_n)}, need {N}")
        samples_n = random.sample(pool_n, N)
        out_s = out_s_pattern.format(label=label)
        with open(out_s, "w", encoding="utf-8") as f:
            for e in samples_n:
                f.write(e["txt"] + "\n")
        print(f"Saved {N} unique transcriptions (len in [{tgt_min},{tgt_max}]) to {out_s}")
        all_used_wavs.update(e["wav"] for e in samples_n)

    # Part 2: single shared ref file (K speakers × E emotions × M utterances)
    usable_entries = entries
    if NEED_NO_OVERLAP:
        usable_entries = [e for e in entries if e["wav"] not in all_used_wavs]
        print(f"Excluded {len(all_used_wavs)} utterances due to NEED_NO_OVERLAP=True")

    spk_emo_dict = {}
    for e in usable_entries:
        if ref_txt_min <= e["len"] <= ref_txt_max:
            spk_emo_dict.setdefault(e["spk"], {}).setdefault(e["emo"], []).append(e)

    valid_speakers = []
    for spk, emo_dict in spk_emo_dict.items():
        valid_emos = [emo for emo, utts in emo_dict.items() if len(utts) >= M]
        if len(valid_emos) >= E:
            valid_speakers.append(spk)

    if len(valid_speakers) < K:
        raise ValueError(
            f"Not enough valid speakers with >= {E} emotions having >= {M} samples "
            f"in ref len [{ref_txt_min},{ref_txt_max}]: have {len(valid_speakers)}, need {K}"
        )

    chosen_spks = random.sample(valid_speakers, K)
    selected = []
    for spk in chosen_spks:
        emo_dict = spk_emo_dict[spk]
        valid_emos = [emo for emo, utts in emo_dict.items() if len(utts) >= M]
        chosen_emos = random.sample(valid_emos, E)
        for emo in chosen_emos:
            selected.extend(random.sample(emo_dict[emo], M))

    with open(out_r, "w", encoding="utf-8") as f:
        for e in selected:
            f.write(e["line"] + "\n")
    print(f"Saved {len(selected)} (K={K}, E={E}, M={M}, ref len [{ref_txt_min},{ref_txt_max}]) wav|transcriptions to {out_r}")


def _build_ref_name_map(ref_dir):
    """Build (spk, emo) → wav filename mapping from a reference folder.

    Scans for files matching spk{spk}_{emo}_ref{r_id}.wav and returns the
    first match per (spk, emo) key.
    """
    mapping = {}
    for fname in os.listdir(ref_dir):
        if not fname.endswith('.wav'):
            continue
        m = re.match(r'spk(\w+)_([A-Za-z]+)_ref\d+\.wav', fname)
        if m:
            key = (m.group(1), m.group(2))
            mapping.setdefault(key, fname)
    return mapping


def make_samples_esd_pair(ref_path, meta_path,
                          out_path="data2/iden_esd_ref_pair.txt",
                          gd_dir=None, ref_dir=None,
                          txt_min=None, seed=42):
    """
    For each utterance in ref_path (one per (speaker, emotion) pair), find a
    different utterance from the same (speaker, emotion) in meta_path and save
    to out_path.  Optionally copy the paired wav files into gd_dir.

    ESD path structure: {root}/{spk}/{emo}/{wavid}.wav
    ref_path  : existing ESD ref file (wav|txt)
    meta_path : full ESD meta file to search for pairs
    out_path  : where to write the pair file
    gd_dir    : if given, copy paired wav files here
    ref_dir   : if given, rename copied files to match the reference folder naming
                (spk{spk}_{emo}_ref{r_id}.wav) so speaker-sim stage can pair them
    txt_min   : optional minimum word count for pair utterances
    """
    random.seed(seed)

    # Load existing ref entries
    ref_entries = []
    used_wavs = set()
    with open(ref_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "|" not in line:
                continue
            wav, txt = line.split("|", 1)
            spk = wav.split("/")[-3]
            emo = wav.split("/")[-2]
            ref_entries.append({"wav": wav, "spk": spk, "emo": emo})
            used_wavs.add(wav)

    # Build per-(spk, emo) pool from meta, excluding used_wavs
    spk_emo_pool = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "|" not in line:
                continue
            wav, txt = line.split("|", 1)
            if wav in used_wavs:
                continue
            spk = wav.split("/")[-3]
            emo = wav.split("/")[-2]
            length = len(txt.split())
            if txt_min is not None and length < txt_min:
                continue
            spk_emo_pool.setdefault((spk, emo), []).append({"wav": wav, "txt": txt, "line": line})

    # For each ref entry, pick a random pair from the same (spk, emo)
    pair_entries = []
    for ref in ref_entries:
        key = (ref["spk"], ref["emo"])
        pool = spk_emo_pool.get(key, [])
        if not pool:
            raise ValueError(f"No pair candidates for speaker {ref['spk']} emotion {ref['emo']} in {meta_path}")
        pair_entries.append(random.choice(pool))

    # Save pair file
    with open(out_path, "w", encoding="utf-8") as f:
        for e in pair_entries:
            f.write(e["line"] + "\n")
    print(f"Saved {len(pair_entries)} ESD pair utterances to {out_path}")

    # Copy wav files to gd_dir
    if gd_dir:
        os.makedirs(gd_dir, exist_ok=True)
        ref_name_map = _build_ref_name_map(ref_dir) if ref_dir else {}
        for ref, pair in zip(ref_entries, pair_entries):
            if ref_name_map:
                dst_name = ref_name_map.get((ref["spk"], ref["emo"]),
                                            os.path.basename(pair["wav"]))
            else:
                dst_name = os.path.basename(pair["wav"])
            shutil.copy2(pair["wav"], os.path.join(gd_dir, dst_name))
        print(f"Copied {len(pair_entries)} wav files to {gd_dir}")


def make_samples_pair(ref_path, meta_path,
                      out_path="data2/iden_libri_ref_pair.txt",
                      gd_dir=None, ref_dir=None,
                      txt_min=None, seed=42):
    """
    For each utterance in ref_path (one per speaker), find a different utterance
    from the same speaker in meta_path and save to out_path.
    Optionally copy the paired wav files into gd_dir.

    ref_path  : existing ref file (wav|txt, one entry per speaker)
    meta_path : full LibriTTS meta file to search for pairs
    out_path  : where to write the pair file
    gd_dir    : if given, copy paired wav files here
    ref_dir   : if given, rename copied files to match the reference folder naming
                (spk{spk}_{emo}_ref{r_id}.wav) so speaker-sim stage can pair them
    txt_min   : optional minimum word count for pair utterances
    """
    random.seed(seed)

    # Load existing ref entries and collect used wav paths + speakers (in order)
    ref_entries = []
    used_wavs = set()
    with open(ref_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "|" not in line:
                continue
            wav, txt = line.split("|", 1)
            spk = wav.split("/")[-3]
            ref_entries.append({"wav": wav, "spk": spk})
            used_wavs.add(wav)

    # Build per-speaker pool from meta, excluding used_wavs
    spk_pool = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "|" not in line:
                continue
            wav, txt = line.split("|", 1)
            if wav in used_wavs:
                continue
            spk = wav.split("/")[-3]
            length = len(txt.split())
            if txt_min is not None and length < txt_min:
                continue
            spk_pool.setdefault(spk, []).append({"wav": wav, "txt": txt, "line": line})

    # For each ref entry, pick a random pair from the same speaker
    pair_entries = []
    for ref in ref_entries:
        spk = ref["spk"]
        pool = spk_pool.get(spk, [])
        if not pool:
            raise ValueError(f"No pair candidates for speaker {spk} in {meta_path}")
        pair_entries.append(random.choice(pool))

    # Save pair file
    with open(out_path, "w", encoding="utf-8") as f:
        for e in pair_entries:
            f.write(e["line"] + "\n")
    print(f"Saved {len(pair_entries)} pair utterances to {out_path}")

    # Copy wav files to gd_dir
    if gd_dir:
        os.makedirs(gd_dir, exist_ok=True)
        # Build spk → ref_filename map from ref_dir (LibriTTS has no emotion dimension)
        spk_ref_map = {}
        if ref_dir:
            full_map = _build_ref_name_map(ref_dir)
            for (spk, emo), fname in full_map.items():
                spk_ref_map.setdefault(spk, fname)
        for ref, pair in zip(ref_entries, pair_entries):
            if spk_ref_map:
                dst_name = spk_ref_map.get(ref["spk"], os.path.basename(pair["wav"]))
            else:
                dst_name = os.path.basename(pair["wav"])
            shutil.copy2(pair["wav"], os.path.join(gd_dir, dst_name))
        print(f"Copied {len(pair_entries)} wav files to {gd_dir}")


if __name__ == '__main__':
    # Libritts: /home/rosen/data/LibriTTS/train-clean-460/724/123284/724_123284_000024_000000.wav|"hˌɔːɹɪzˈɔntəli, θɹˈiː hˈʌndɹɪd ænd fˈɪfti lˈiːɡz fɹʌm ˈaɪslənd."|724
    # ESD: /hdd/ESD/0019/Sad/evaluation/0019_001061.wav|Chapter ten a warm welcome.

    # LibriTTS for preservation ()
    meta_file = "/home/rosen/data/LibriTTS/libri_clean_460.txt"  #
    meta_file_esd = "/home/rosen/data/ESD_24k/esd.txt"

    RANDOM_LIBRI, FINE_LIBRI, SPK_LIBRI, PAIR_LIBRI = False, False, False, True

    RANDOM_ESD, FINE_ESD, SPK_ESD, PAIR_ESD = False, False, False, True
    if RANDOM_LIBRI:
        make_samples(meta_file, N=10, K=4, M=5, out_s="data/libri_s1.txt", out_r="data2/libri_r1.txt", NEED_NO_OVERLAP=True)
    if FINE_LIBRI:  # NOT used
        make_samples_lengthratio(
            meta_file, N=2, K=4, M=10,
            tgt_ranges=[("ratio05", 4, 8), ("ratio1", 8, 12), ("ratio2", 14, 20)],
            ref_txt_min=8, ref_txt_max=12,
            out_s_pattern="data2/fine_libri_syn_{label}.txt",
            out_r="data2/fine_libri_ref.txt",
            NEED_NO_OVERLAP=True)
    if SPK_LIBRI:
        iden_sen_num, iden_spk_num, iden_ref_num = 10, 50, 1
        make_samples(
            meta_file,
            N=iden_sen_num, K=iden_spk_num, M=iden_ref_num,
            out_s="data2/iden_libri_syn.txt", out_r="data2/iden_libri_ref.txt", txt_min=3)
    if PAIR_LIBRI:
        make_samples_pair(
            ref_path="data2/iden_libri_ref.txt",
            meta_path=meta_file,
            out_path="data2/iden_libri_ref_pair.txt",
            gd_dir="/home/rosen/ckpt/exp2/iden_libritts/gd/",
            ref_dir="/home/rosen/ckpt/exp2/iden_libritts/reference/random/",
            txt_min=3)

    # ESD for base dtw preservation

    # ESD for fine-grained dtw preservation
    if FINE_ESD:
        fine_sen_num, fine_spk_num, fine_emo_num, fine_ref_num = 10, 2, 5, 2
        make_samples_esd_lengthratio(
            meta_file_esd,
            N=fine_sen_num, K=fine_spk_num, E=fine_emo_num, M=fine_ref_num,
            tgt_ranges=[("ratio05", 3, 5), ("ratio1", 6, 9), ("ratio2", 10, 12)],
            ref_txt_min=6, ref_txt_max=9,
            out_s_pattern="data2/fine_esd_syn_{label}.txt",
            out_r="data2/fine_esd_ref.txt",
            NEED_NO_OVERLAP=True)
    # ESD for speaker identity
    if SPK_ESD:
        iden_sen_num, iden_spk_num, iden_emo_num, iden_ref_num = 10, 10, 5, 1
        make_samples_esd(
            "/home/rosen/data/ESD_24k/esd.txt",
            N=iden_sen_num, K=iden_spk_num, E=iden_emo_num, M=iden_ref_num,
            out_s="data2/iden_esd_syn.txt", out_r="data2/iden_esd_ref.txt"
        )
    if PAIR_ESD:
        make_samples_esd_pair(
            ref_path="data2/iden_esd_ref.txt",
            meta_path=meta_file_esd,
            out_path="data2/iden_esd_ref_pair.txt",
            gd_dir="/home/rosen/ckpt/exp2/iden_esd/gd/",
            ref_dir="/home/rosen/ckpt/exp2/iden_esd/reference/random/"
        )

"""
Create evaluation dataset, including two types data: syn.txt and ref.txt, for random, fine test. 
syn.txt is target text file to be synthesized.
ref.txt is reference speech file (formated as path/txt) used as reference.

We already have creation code on libritts dataset for random, fine test, which are make_samples and make_samples_lengthratio.
1. In make_samples (random test), we randomly chosen N synthesis text (saved in s1.txt), and K speaker * M utterances/speaker on libritts (saved in r1.txt), 
2. In make_samples_lengthratio (fine test), same as 1, but the length of synthesis txt and renference txt are constrained by the tgt_txt_min/max, ref_txt_min/max parameters.

Now, write the script to create syn.txt and ref.txt on esd dataset for random and fine test. require
1. ESD Input: "/home/rosen/data/ESD_24k/esd.txt". (LibriTTS input: "/home/rosen/data/LibriTTS_16k/libri_tts_360.txt")
2. for ramdom test, randomly chosen N synthesis text (save in random_s_esd.txt), and K speakers * E emotions * M utterances/speaker&emotion (saved in random_r_esd.txt)
3. for fine test, same as 2, but the length of synthesis txt and renference txt are constrained by the tgt_txt_min/max, ref_txt_min/max parameters.
"""