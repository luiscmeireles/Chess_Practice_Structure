#!/usr/bin/env bash
set -e
mkdir -p data

echo "⬇️ Downloading dataset from Zenodo..."
wget -O data/df_final.feather "https://zenodo.org/records/17122889/files/df_final.feather?download=1"
echo "✅ Saved to data/df_final.feather"
