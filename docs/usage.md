# CIC – Retrieval Pipeline (Full Usage Guide)
This document describes the full experimental pipeline used to evaluate the Cromelin Information Compiler (CIC) as the NOS-native waveform compiler for information retrieval.
The goal is simple reproducibility:
- Load the judged subset of TREC DL 2019
- Convert documents into CIC complex waveforms
- Encode queries
- Perform full-scan resonance scoring
- Compute metrics (MRR, nDCG, Recall)

## Contents
- collection.tsv — Document collection  
- queries.tsv — Evaluation queries  
- qrels.txt — Human-judged relevance labels  
- trec_dl2019_<timestamp>.txt — Evaluation logs  
- make_trec_data.py — Generates the DL2019 subset  
- evaluation/ — Evaluation runner + metric computation  
- encoders/ — Text → CIC waveform encoders  
- store/ — Memory store  
- logs/ — Auto-generated runtime logs  
- main.py — Example entry point

## Environment Setup
Requirements:
- Python 3.10+
- Windows / PowerShell recommended
- Virtual environment recommended

Create venv:
python -m venv .venv
.\.venv\Scripts\Activate.ps1

Install dependencies:
pip install ir_datasets numpy requests sentence-transformers

## Preparing the Dataset
Generate the TREC DL 2019 judged slice:
python -X utf8 make_trec_data.py

Creates:
data/
  collection.tsv
  queries.tsv
  qrels.txt

## Changing Dataset Size
Inside make_trec_data.py, modify:
target_size: int = 1_000_000

## Running Evaluation
Set project root (PowerShell):
$env:PYTHONPATH = "E:\NOS-IR"

Semantic Embedding → CIC Waveforms:
python -m evaluation.runner `
  --collection data\collection.tsv `
  --queries data\queries.tsv `
  --qrels data\qrels.txt `
  --encoder embed `
  --model all-MiniLM-L6-v2 `
  --N 512 `
  --topk 100

Character Wave Baseline:
python -m evaluation.runner `
  --collection data\collection.tsv `
  --queries data\queries.tsv `
  --qrels data\qrels.txt `
  --encoder char `
  --N 512 `
  --topk 100

Logs saved to:
logs/

## Debugging and Development
Inspect waveform generation:
python encoders/char_wave.py

Kill stray Python processes:
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force

## Notes
- --encoder embed → MiniLM semantic embeddings → CIC waveform  
- --encoder char → character-level waveform baseline  
- CIC scoring uses phase-geometric resonance across top-K FFT bins  
- Evaluations use the judged TREC DL 2019 slice

