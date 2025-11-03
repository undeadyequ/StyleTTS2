def extend_phone2syl(phones, phone_durs, syl_start_index=None, intermark="11"):
    """
    Args:
        phones: ["", "AH", "", 11, "B", "", "AH0", "", "IH, "", 11,"", "AH", "", "G", ""]
        phone_durs: [2, 3, 5, 6, ...]
    Returns:
        syls: [" AH", " 11"]
        syl_durs  [5, 11, ..., ...]
    """
    assert len(phones) == len(phone_durs)
    # get start index of each syllables
    if syl_start_index is None:
        syl_start_index = search_syllable_index(phones, "ipa")  # [0, 3, 7, 10]
    syls = []
    syl_durs = []

    # get syllables and its duration
    phones = [str(phone).replace(intermark, "") for phone in phones]  # do not show 11
    for i in range(len(syl_start_index)):
        syl_start = syl_start_index[i]
        if i != len(syl_start_index) - 1:
            syl_end = syl_start_index[i+1]
        else:
            syl_end = len(phones)           # last syllables includes to the last phone
        syl_durs.append(sum(phone_durs[syl_start : syl_end]))
        syls.append(" ".join(phones[syl_start : syl_end]))
    return syls, syl_durs


def search_syllable_index(phones, syllabel_system="cmu"):
    """
    phone must include space (11) and interval ("")
    Args:
        phones:  ["", "AH", "", "B", "11", "", "G"]   #  w_space: 11, p_space: ""

    Syllable = (consonant) + vowel + (consonant)

    if words has multiple vowels
        s1 = (consonant) + vowel
        s2 = (consonant) + vowel
        s_last = ()


    Returns:

    """
    symbols = {"a": "ə", "ey": "eɪ", "aa": "ɑ", "ae": "æ", "ah": "ə", "ao": "ɔ",
               "aw": "aʊ", "ay": "aɪ", "ch": "ʧ", "dh": "ð", "eh": "ɛ", "er": "ər",
               "hh": "h", "ih": "ɪ", "jh": "ʤ", "ng": "ŋ",  "ow": "oʊ", "oy": "ɔɪ",
               "sh": "ʃ", "th": "θ", "uh": "ʊ", "uw": "u", "zh": "ʒ", "iy": "i", "y": "j"}

    vowel = ["ao", "uw", "eh", "ah", "aa", "iy", "ih", "uh", "ae", "aw", "ay", "er", "ey", "ow", "oy"]
    # vowel = ["AO", "UW", "EH", "AH", "AA", "IY", "IH", "UH", "AE", "AW", "AY", "ER", "EY", "OW", "OY"]

    if syllabel_system == "ipa":
        vowel = [symbols[v] for v in vowel]

    # Cluster phones to words
    phones_by_word = []         # [["", "AH", ""], ["11", "B", "", "AH0", "", "IH, """], ["11","", "AH", "", "G", ""]]
    start = 0
    phones = [str(phone) for phone in phones]
    for i, phone in enumerate(phones):
        if phone == "11":
            phones_by_word.append(phones[start:i])
            start = i
        if i == len(phones) - 1:
            phones_by_word.append(phones[start:])

    # split words to syllabels by vowel
    phones_by_syllabel = []      # [["", "AH", ""], ["11", "B", "", "AH0"], ["", "IH, ""], ["11","", "AH", "", "G", ""]]
    for i, phones in enumerate(phones_by_word):
        ## remove digits
        phones_rm_digit = []    # ["11", "B", "", "AH0", "", "IH, """] -> [..., "AH", ...]
        for phone in phones:
            if len(phone) > 0:
                if phone[-1].isdigit():
                    phones_rm_digit.append(phone[:-1])
                    continue
            phones_rm_digit.append(phone)
        ## index vowel
        vowel_indexs = [i for i, p in enumerate(phones_rm_digit) if p in vowel]  # -> [3, 5]
        ## split to syllable by vowel
        if len(vowel_indexs) <= 1:                           # words with single or no vowel (alphabet, exp: f o r e t !)
            phones_by_syllabel.append(phones)
        else:                                                # words with multiple vowels
            start = 0
            for j, v_ind in enumerate(vowel_indexs):
                phones_by_syllabel.append(phones[start:v_ind+1])  # Syllable = (consonant) + vowel
                start = v_ind + 1
            if vowel_indexs[-1] < len(phones) - 1:    # if words not end with vowel, add remain consonant
                phones_by_syllabel[-1].extend(phones[vowel_indexs[-1]+1:])  # Last Syl = (consonant) + vowel + (consonant)

    syllabel_len = [len(phones) for phones in phones_by_syllabel]
    syllabel_index = [sum(syllabel_len[:i]) for i in range(len(syllabel_len))]    # [0, 3, 7, 10]

    return syllabel_index