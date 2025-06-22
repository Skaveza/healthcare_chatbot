import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
import os
from sklearn.utils import resample

def balance_classes(df, max_samples_per_class=30):
    grouped = df.groupby('focus_area')
    balanced = []
    for name, group in grouped:
        if len(group) > max_samples_per_class:
            balanced.append(group.sample(max_samples_per_class, random_state=42))
        else:
            balanced.append(group)
    return pd.concat(balanced).sample(frac=1, random_state=42)

def medical_split(df):
    df = df.dropna(subset=['question', 'answer', 'focus_area'])
    df = df[df['focus_area'] != 'UNKNOWN']
    
    def get_question_type(q):
        q = q.lower()
        if 'what is' in q or 'define' in q: return 'DEF'
        elif 'symptom' in q or 'sign' in q: return 'SX'
        elif 'treat' in q or 'therapy' in q: return 'TX'
        elif 'diagnos' in q or 'test for' in q: return 'DX'
        else: return 'CLS'
    
    df['q_type'] = df['question'].apply(get_question_type)
    
    train, temp = train_test_split(
        df,
        test_size=0.3,
        stratify=df['q_type'],
        random_state=42
    )
    
    val, test = train_test_split(
        temp,
        test_size=0.5,
        stratify=temp['q_type'],
        random_state=42
    )
    
    print("\n=== Balanced Splits ===")
    for name, split in [('TRAIN', train), ('VAL', val), ('TEST', test)]:
        print(f"\n{name}: {len(split)} samples")
        print(split['q_type'].value_counts())
    
    return train, val, test

if __name__ == "__main__":
    df = pd.read_csv('/kaggle/input/medquad-csv/medquad.csv')
    train_df, val_df, test_df = medical_split(df)
    
    os.makedirs("/kaggle/working/splits", exist_ok=True)
    train_df.to_csv("/kaggle/working/splits/train.csv", index=False)
    val_df.to_csv("/kaggle/working/splits/val.csv", index=False)
    test_df.to_csv("/kaggle/working/splits/test.csv", index=False)
    print("\nSplits saved!")
