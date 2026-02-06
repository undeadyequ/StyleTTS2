import torch
import torch.nn.functional as F
from utils import make_attn_f2p_from_durations, extract_pitch_trend_v2, extract_pitch_trend, _interp_unvoiced_1d
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_extract_pitch_trend_logf0(device="cpu"):
    torch.manual_seed(0)
    B, Tp = 2, 6

    # durations -> Tf (keep Tf reasonably large for visualization; but still robust)
    durs = torch.randint(6, 12, (B, Tp), device=device)
    durs[1] = durs[0]
    Tf = int(durs[0].sum().item())

    attn_f2p = make_attn_f2p_from_durations(durs)  # [B,Tp,Tf]

    # synthetic pitch (Hz)
    t = torch.linspace(0, 1, Tf, device=device)
    pitch = 180.0 + 40.0 * torch.sin(2 * torch.pi * 2 * t)  # [Tf]
    pitch = pitch.unsqueeze(0).repeat(B, 1)                 # [B,Tf]

    # Insert "unvoiced junk" segments safely within Tf
    def set_junk(start: int, length: int, max_hz: float):
        start = max(0, min(start, Tf))
        end = max(start, min(start + length, Tf))
        if end > start:
            pitch[:, start:end] = torch.rand(B, end - start, device=device) * max_hz

    # two junk regions (sizes auto-adjust if Tf is small)
    set_junk(start=Tf // 4, length=max(2, Tf // 10), max_hz=3.0)
    set_junk(start=Tf * 3 // 4, length=max(2, Tf // 12), max_hz=15.0)

    tareget_len = Tp
    log_trend = extract_pitch_trend(pitch, attn_f2p, tareget_len)

    print("Tf =", Tf)
    print("pitch:", pitch.shape)
    print("attn_f2p:", attn_f2p.shape)
    print("log_trend:", log_trend.shape)

    assert log_trend.shape == (B, 1, tareget_len)
    assert torch.isfinite(log_trend).all()
    print("OK")


def test_extract_pitch_trend_inference():
    # 1. Setup dimensions
    B = 2
    Tp_gd, Tf_gd = 4, 100  # Reference: 4 phonemes, 100 frames
    Tp_syn, Tf_syn = 8, 150  # Target: 8 phonemes, 150 frames
    tgt_len = 128  # Final BERT embedding length

    # 2. Mock Pitch Ground Truth (Reference Audio)
    # Let's create a simple rising pitch from 100Hz to 200Hz
    pitch_gd_frame = torch.linspace(100, 200, Tf_gd).repeat(B, 1)

    # 3. Mock Ground Truth Attention [B, Tp, Tf]
    # Simple linear alignment: each phoneme gets 25 frames
    attn_f2p_gd = torch.zeros(B, Tp_gd, Tf_gd)
    for i in range(Tp_gd):
        attn_f2p_gd[:, i, i * 25:(i + 1) * 25] = 1.0

    # 4. Mock Predicted Attention [B, Tp_syn, Tf_syn]
    # Target has 8 phonemes, each getting ~18-19 frames
    attn_f2p_pred = torch.zeros(B, Tp_syn, Tf_syn)
    step = Tf_syn // Tp_syn
    for i in range(Tp_syn):
        attn_f2p_pred[:, i, i * step:(i + 1) * step] = 1.0

    print(f"Input Shape: {pitch_gd_frame.shape}")
    print(f"GD Phonemes: {Tp_gd}, Pred Phonemes: {Tp_syn}")

    # 5. Run Function
    try:
        trend = extract_pitch_trend_v2(
            pitch_gd_frame=pitch_gd_frame,
            attn_f2p_gd=attn_f2p_gd,
            tgt_frame_len=tgt_len,
            attn_f2p_pred=attn_f2p_pred,
            fmin=50.0,
            fmax=600.0
        )
        print("trend:", trend[0,0,:100])

        # 6. Validations
        assert trend.shape == (B, 1, tgt_len), f"Wrong output shape: {trend.shape}"
        assert torch.isfinite(trend).all(), "Trend contains NaNs or Infs"

        # Check if the pitch is still in a reasonable log range
        # log(100) approx 4.6, log(200) approx 5.3
        assert trend.mean() > 4.0 and trend.mean() < 6.0, f"Unreasonable log-pitch: {trend.mean()}"

        print("✅ Test Passed: Shape and values are correct.")
        print(f"Output Shape: {trend.shape}")

    except Exception as e:
        print(f"❌ Test Failed: {str(e)}")
        raise e

if __name__ == '__main__':
    test_extract_pitch_trend_inference()