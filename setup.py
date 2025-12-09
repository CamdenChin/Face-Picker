#!/usr/bin/env python3
"""
Simple Setup Script
"""

import os
import glob
import pandas as pd
import subprocess
import sys
import importlib


# ----------------------------------------------------
# Install dependencies if missing
# ----------------------------------------------------

REQUIRED = [
    "pandas",
    "torch",
    "torchvision",
    "pillow",
    "matplotlib",
    "tqdm"
]

def install_deps():
    print("Checking dependencies (installing if missing)...")
    for pkg in REQUIRED:
        try:
            importlib.import_module(pkg)
            print(f"  ✓ {pkg} is already installed")
        except ImportError:
            print(f"  ⬇ Installing {pkg} ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])



# ----------------------------------------------------
# MAIN SCRIPT
# ----------------------------------------------------

def main():

    # step 0 — make sure deps are installed
    install_deps()

    csv_files = glob.glob('*.csv')
    
    if not csv_files:
        print("❌ No CSV files found in current directory!")
        return
    
    print(f"Found {len(csv_files)} CSV file(s):")
    for f in csv_files:
        print(f"  - {f}")
    
    target_csv = None
    
    for pattern in ['rescaled', 'gender', '2025-12-03_2']:
        matches = [f for f in csv_files if pattern in f.lower()]
        if matches:
            target_csv = matches[0]
            break
    
    if not target_csv:
        target_csv = csv_files[0]
    
    print(f"\n✓ Using: {target_csv}")
    with open(target_csv, 'r') as f:
        first_line = f.readline()
        delimiter = ';' if ';' in first_line else ','
    
    print(f"✓ Delimiter: '{delimiter}'")
    
    try:
        df = pd.read_csv(target_csv, sep=delimiter)
        print(f"✓ Loaded {len(df)} rows")
        print(f"✓ Columns: {list(df.columns)[:5]}...")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return
    
    print("\n🚻 Step 3: Checking for gender data...")
    
    has_gender = 'gender' in df.columns or 'sex' in df.columns
    
    if has_gender:
        gender_col = 'gender' if 'gender' in df.columns else 'sex'
        print(f"✓ Gender column found: '{gender_col}'")
        print(f"\nGender distribution:")
        print(df[gender_col].value_counts())
    else:
        print("❌ No gender column found!")
        print("\nYour CSV needs a 'gender' or 'sex' column")
        print("Please add it in Excel/Numbers and re-upload")
        return
        
    scores = df['label']
    print(f"\nScore statistics:")
    print(f"  Mean:   {scores.mean():.2f}")
    print(f"  Median: {scores.median():.2f}")
    print(f"  Min:    {scores.min():.2f}")
    print(f"  Max:    {scores.max():.2f}")
    print(f"  Std:    {scores.std():.2f}")
    
    needs_rescaling = scores.mean() < 4.0 or scores.std() < 2.0
    
    if needs_rescaling:
        print(f"   Mean score is {scores.mean():.2f} (should be ~5.5)")
        print("   Scores are compressed - model will predict too low")
        print("\n✓ SOLUTION: Need to rescale scores to 1-10 range")
        
        df['percentile'] = df['label'].rank(pct=True)
        df['label'] = df['percentile'] * 9 + 1
        df.drop('percentile', axis=1, inplace=True)
        
        output_file = 'normalized_celebrity_ratings_rescaled.csv'
        df.to_csv(output_file, index=False, sep=delimiter)
        
        print(f"✓ Saved rescaled data: {output_file}")
        print(f"\n  New mean: {df['label'].mean():.2f}")
        print(f"  New std:  {df['label'].std():.2f}")
        print(f"  Range:    {df['label'].min():.2f} - {df['label'].max():.2f}")
        
        print("\n Ready for training!")
        
    else:
        print("  No rescaling needed")
    
    print("\n" + "="*70)
    print("  SETUP COMPLETE!")
    print("="*70)
    
    if needs_rescaling:
        print(f"\n✓ Fixed dataset: {output_file}")
    else:
        print(f"\n✓ Dataset ready: {target_csv}")
    

    print(f"  Total celebrities: {len(df)}")
    male_count = len(df[df[gender_col] == 'male'])
    female_count = len(df[df[gender_col] == 'female'])
    print(f"  Males:   {male_count}")
    print(f"  Females: {female_count}")
    print(f"  Score range: {df['label'].min():.1f} - {df['label'].max():.1f}")
    
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
