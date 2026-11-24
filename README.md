# Gemini Decisions API vs LLM: a reproducible test

Google's [Gemini Decisions API](https://aistudio.google.com/docs/decisions-api) can't write anything. It only picks from the options you give it, and tells you how sure it is. This repository tests it against normal LLMs on the job it's built for: sorting a business inbox of **emails, photos, screen recordings and PDFs**.

**Read more:** the full write-up, with the diagrams, the hard cases one by one and how to set it up, is on the ReachMind blog: **[Gemini Decisions API vs LLM: cost, speed and accuracy, tested](https://www.reachmind.co/blog/decisions-api-vs-llm/)**.

## Results in one line

On this test, **Gemini Decision 1 was 3 to 7 times cheaper and about twice as fast as Gemini 3.5 Flash-Lite, but not more accurate.** It gave the same answer in every run, and when it was at least 90% sure it was almost always right, but most answers didn't reach that bar.

Tested on 2026-11-24 from Japan, with google-genai 3.6.1. Total API cost of the whole run: **$0.17**.

## The four lanes

Every item gets the same 8 questions (3 yes/no, 3 choice, 2 score), with the same wording and option order, in one call per item.

| Lane | What it does |
|---|---|
| A. Cheap LLM | **Gemini 3.5 Flash-Lite** reads the media directly, structured output with an enum per question, thinking `minimal`, temperature 0 |
| B. Stronger LLM | **Gemini 3.8 Flash**, same setup, thinking `low` (its lowest). Reference only: run once |
| C. Convert first | OCR, video frames every 2 s and a one-paragraph description turn the item into text, then **Gemini 3.5 Flash-Lite** answers from text only. This is what a text-only decision model needs |
| D. Decisions API | **Gemini Decision 1** reads the media directly, `resolution: low` |

## Results: Kumo Supply inbox (120 decisions, 3 runs)

15 synthetic items for a fictional packaging supplier in Tokyo: 4 emails and chats, 3 photos and screenshots, 4 screen recordings, 4 PDFs. The correct answers are known by construction (see `labels.yaml`).

| Type | Model | Cost per 1,000 items | Median time per item | Accuracy (strict / lenient) | Same answer in all runs |
|---|---|---|---|---|---|
| Emails | Gemini 3.5 Flash-Lite | $0.39 | 0.84s | 97% / 97% | 100% |
| Emails | Gemini 3.8 Flash | $1.01 | 3.41s | 94% / 97% | 1 run |
| Emails | Gemini Decision 1 | $0.08 | 0.47s | 94% / 97% | 100% |
| Photos | Gemini 3.5 Flash-Lite | $0.69 | 1.31s | 79% / 88% | 100% |
| Photos | Gemini 3.8 Flash | $1.87 | 3.61s | 96% / 100% | 1 run |
| Photos | Flash-Lite on converted text | $1.42 | 3.02s | 83% / 90% | 83% |
| Photos | Gemini Decision 1 | $0.10 | 0.52s | 83% / 88% | 100% |
| Recordings | Gemini 3.5 Flash-Lite | $0.69 | 1.63s | 90% / 98% | 94% |
| Recordings | Gemini 3.8 Flash | $2.56 | 7.99s | 91% / 100% | 1 run |
| Recordings | Flash-Lite on converted text | $3.82 | 7.83s | 90% / 97% | 91% |
| Recordings | Gemini Decision 1 | $0.19 | 0.94s | 84% / 97% | 100% |
| PDFs | Gemini 3.5 Flash-Lite | $0.59 | 1.26s | 98% / 98% | 97% |
| PDFs | Gemini 3.8 Flash | $2.65 | 7.50s | 97% / 97% | 1 run |
| PDFs | Flash-Lite on converted text | $0.47 | 1.00s | 95% / 95% | 97% |
| PDFs | Gemini Decision 1 | $0.19 | 0.62s | 94% / 97% | 100% |

Strict: the answer matches the label. Lenient: it matches the label or a defensible alternative listed in `labels.yaml`. Times include the network round trip from Japan and, for lane C, the conversion.

At **medium** resolution instead of low, Gemini Decision 1 scored 83% on photos and 83% on recordings: no improvement.

### When Gemini Decision 1 says it's sure

| Type | Answers with confidence ≥ 0.9 | Accuracy on those answers |
|---|---|---|
| Emails | 69% | 96% |
| Photos | 42% | 100% |
| Recordings | 28% | 89% |
| PDFs | 52% | 100% |

It was confidently wrong (≥ 0.9) on two questions, the same way in all 3 runs: a quote request with a name and a work email marked "no personal data", and a tracking dashboard with 37 delayed parcels marked "shows no error" (defensible: the warning isn't a software error). Full list in `results/2026-11-24/confident_wrong.csv`.

## Results: 10 messy public items (40 decisions, 3 runs)

| Items | Model | Accuracy | Cost per 1,000 items | Median time |
|---|---|---|---|---|
| 5 customer messages (Bitext) | Gemini 3.5 Flash-Lite | 95% | $0.18 | 0.78s |
| 5 customer messages (Bitext) | Gemini Decision 1 | 85% | $0.04 | 0.38s |
| 5 receipt photos (CORD) | Gemini 3.5 Flash-Lite | 100% | $0.51 | 1.29s |
| 5 receipt photos (CORD) | Gemini Decision 1 | 95% | $0.06 | 0.53s |

## Reproduce it

You need Python 3.12+, [uv](https://docs.astral.sh/uv/), `ffmpeg`, `tesseract`, and a Gemini API key from a Google Cloud project **with billing on**. On the free tier Google may use your data, and its rate limits distort the timings. The full run cost us under $0.20.

1. **Clone and install**
   ```bash
   git clone https://github.com/paulomarcos/gemini-decisions-api-vs-llm.git
   cd gemini-decisions-api-vs-llm/harness
   uv sync
   ```
2. **OCR language data** for lane C (English and Japanese):
   ```bash
   mkdir -p tessdata
   curl -L -o tessdata/eng.traineddata https://github.com/tesseract-ocr/tessdata_fast/raw/main/eng.traineddata
   curl -L -o tessdata/jpn.traineddata https://github.com/tesseract-ocr/tessdata_fast/raw/main/jpn.traineddata
   ```
3. **Your API key**, in `harness/.env` (git-ignored, never commit it):
   ```bash
   echo "GEMINI_API_KEY=your-key-here" > .env
   ```
4. **Check prices and models** in `harness/config.yaml` against the [Gemini pricing page](https://aistudio.google.com/docs/pricing). They change fast.
5. **Fetch the public items** (downloaded to `public/cache/`, not stored in this repo, so each dataset keeps its licence):
   ```bash
   cd ../public && uv run --project ../harness python fetch_public.py messages receipts && cd ../harness
   ```
   Then `uv run --project ../harness python fetch_public.py compose` from `public/` if you change the picks.
6. **Dry run** (no API calls, fake answers, checks the plumbing):
   ```bash
   uv run python run.py --dry-run
   ```
7. **The real run**: 3 runs, all lanes, both tests. It stops on its own if the projected cost after run 1 goes over $1, or if the spend reaches $0.95.
   ```bash
   uv run python run.py --confirm-paid
   uv run python summarize.py ../results/decisions-test-YYYY-MM-DD
   ```
8. **Optional:** re-run lane D on photos and recordings at medium resolution:
   ```bash
   uv run python run.py --confirm-paid --test kumo --lanes D --types image,video --resolution medium --tag medium
   ```

To rebuild the synthetic items from their templates (no generative AI involved): `cd build && python3 build_items.py ../kumo` (needs Playwright with Chrome, OpenCV, Pillow and ffmpeg).

## What's in here

```
kumo/                 the 15 synthetic inbox items (text, image, video, pdf)
questions.yaml        the 8 questions per item, identical for every lane
labels.yaml           the correct answers, with defensible alternatives (also_ok)
build/                regenerates every synthetic item from templates
harness/              run.py (the lanes and the run), summarize.py, config.yaml (models, prices, budget)
public/               the second test: fetch script, questions, item IDs and labels
results/2026-11-24/   raw.jsonl (one row per answer), summary.csv, hard_cases.csv, confident_wrong.csv, cover_chart.png
```

## Limits

- 120 + 40 decisions is a small sample, not a benchmark.
- The synthetic items and their answers were written by the author. That's why the public items exist.
- The public items cover messages and receipt photos only. Recordings, contracts and parcel photos with a usable licence weren't available without paid accounts.
- Urgency labels on the public messages were made by a model and checked with a keyword rule (`public/label_check.md`).
- Screen recordings only: no camera footage, no call audio. One region (Japan), one day.
- Item I3 (a damaged-parcel photo) is in `labels.yaml` but has no file, so it is skipped.

## Licence

Code and the synthetic Kumo Supply items: MIT. The public datasets are not redistributed here: [CORD](https://github.com/clovaai/cord) is CC BY 4.0 and the [Bitext customer support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) is CDLA-Sharing 1.0.

Made by [Paulo Dores](https://www.linkedin.com/in/paulo-dores) at [ReachMind](https://www.reachmind.co).
