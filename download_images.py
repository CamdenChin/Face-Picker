import pandas as pd
import requests
from pathlib import Path

csv_file = "normalized_celebrity_ratings_rescaled.csv"
df = pd.read_csv(csv_file, sep=';')

outdir = Path("celebrity_images")
outdir.mkdir(exist_ok=True)

for _, row in df.iterrows():
    url = row["image_path"]
    img_id = row["id"]
    outfile = outdir / f"{img_id}.jpg"
    
    if outfile.exists():
        continue

    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            with open(outfile, "wb") as f:
                f.write(r.content)
            print(f"Downloaded {outfile}")
        else:
            print(f"Failed {url}")
    except Exception as e:
        print(f"Error {url}: {e}")
