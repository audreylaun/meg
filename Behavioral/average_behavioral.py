import pandas as pd
import os

subs = ['R3250', 'R3254', 'R3261', 'R3264', 'R3270', 'R3271', 'R3272','R3275',
        'R3277', 'R3279', 'R3285','R3286', 'R3289', 'R3290', 'R3326', 'R3327', 'R3328']

conditions = ['B','C','D','A','D','A','B','D','B','C','E','F','G','H','E','F','G']

directory = '/Users/audreylaun/Library/CloudStorage/Box-Box/Starling/Experiment1/Behavioral_Data/'
master_file = os.path.join(directory, 'RTs_and_Accuracy_ALL.xlsx')

master_df = pd.read_excel(master_file)

for i, sub in enumerate(subs):
    condition = conditions[i]

    # --- Map conditions to prod file ---
    if condition in ['A', 'D']:
        prod_condition = 'B1'
    elif condition in ['B', 'C']:
        prod_condition = 'A1'
    elif condition in ['E', 'H']:
        prod_condition = 'B2'
    elif condition in ['F', 'G']:
        prod_condition = 'A2'
    else:
        raise ValueError(f"Unexpected condition {condition} for subject {sub}")

    file = os.path.join(directory, f'RTs_{prod_condition}.csv')
    df = pd.read_csv(file)

    print(df.head())
    print(type(sub))

    acc_col = f"{sub} Accuracy"
    rt_col = f"{sub} RTs"

    # Convert to numeric
    df[acc_col] = pd.to_numeric(df[acc_col], errors="coerce")
    df[rt_col] = pd.to_numeric(df[rt_col], errors="coerce")

    # Accuracy
    accuracy_count = df.loc[df["Triggers"] != 136, acc_col].sum()

    # Identical RTs (Trigger 130, correct only)
    mask_ident = (df["Triggers"] == 130) & (df[acc_col] == 1)
    ident_rts = df.loc[mask_ident, rt_col]
    ident_avg = ident_rts.mean()
    ident_sd = ident_rts.std()

    # Unrelated RTs (Trigger 132, correct only)
    mask_unrel = (df["Triggers"] == 132) & (df[acc_col] == 1)
    unrel_rts = df.loc[mask_unrel, rt_col]
    unrel_avg = unrel_rts.mean()
    unrel_sd = unrel_rts.std()

    row = {
        "Subject": sub,
        "Accuracy": accuracy_count,
        "Identical RTs Avg": ident_avg,
        "Identical RTs SD": ident_sd,
        "Unrelated RTs Avg": unrel_avg,
        "Unrelated RTs SD": unrel_sd
    }
    master_df = pd.concat([master_df, pd.DataFrame([row])], ignore_index=True)

master_df.to_excel(master_file, index=False)

