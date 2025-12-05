import pandas as pd

def compute_mean_df_over_model(file_path):
    df = pd.read_csv(file_path)

    # Drop the first two header rows
    df_clean = df.drop([0, 1]).reset_index(drop=True)

    # Fix model column name
    df_clean = df_clean.rename(columns={'Unnamed: 0': 'model'})

    # Convert numeric values
    numeric_cols = df_clean.columns.drop('model')
    df_clean[numeric_cols] = df_clean[numeric_cols].apply(pd.to_numeric)

    # Identify pitch and energy columns
    emotion_pitch_cols = [c for c in df_clean.columns if not c.endswith('.1') and c != 'model']
    emotion_energy_cols = [c for c in df_clean.columns if c.endswith('.1')]

    # Compute per-model averages
    df_clean['pitch'] = df_clean[emotion_pitch_cols].mean(axis=1).round(2)
    df_clean['energy'] = df_clean[emotion_energy_cols].mean(axis=1).round(2)

    result = df_clean[['model', 'pitch', 'energy']].set_index('model')

    return result

if __name__ == '__main__':
    df_csv = "/home/rosen/ckpt/exp/mdit_tts_esd/statsIntp_psd_mulitindex_cmpM.csv"
    df_mean_csv = "/home/rosen/ckpt/exp/mdit_tts_esd/statsIntp_psd_mulitindex_cmpM_meanModel.csv"
    df_mean = compute_mean_df_over_model(df_csv)
    df_mean.to_csv(df_mean_csv, index_label=True)