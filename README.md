# ISYS1079 Undergraduate Assignment 2 Starter

This starter runs B0 before you change any code.

## Where to Start

The main process is at the top of each stage file:

| Stage | File and function |
| --- | --- |
| 1. Build the index | `index/build.py`: `build_resources` |
| 2. Retrieve and rank | `rag_survey/retrieve.py`: `run_retrieval` |
| 3. Judge and save results | `rag_survey/judge.py`: `run_judging` |

Start your retrieval method in **`retrieval/rankers.py`** and your judging method in **`judging/judges.py`**.

| Folder | Provided code |
| --- | --- |
| `retrieval/` | `baseline.py` contains B0 retrieval. `vectors.py` loads aligned document vectors. |
| `judging/` | `baseline.py` contains B0 option matching. `aggregate.py` converts judgments to probabilities. |
| `index/` | `access.py` opens the index and reads documents. `text_processing.py` constructs and tokenises text. |
| `api/` | `chat.py` calls a chat model and caches raw replies. `embeddings.py` encodes text. |
| `common/` | Config reading, file reading/writing, and output checks. |
| `metrics/`, `test_sanity/` | Public evaluation and mini self-check. |

The Main Specification defines the assessed work. This README is for setup and running the code.

To follow a run, read the numbered steps in the stage functions above, then the selected method branch. Calls to `common/validation.py` check return formats and raise an error before saving invalid records. The small `main()` functions read command-line arguments and start their stage.

## Setup

1. Extract the starter ZIP and open this folder itself in VS Code.
2. Create and select a local `.venv` using Python 3.10 to 3.14 inclusive.
3. Install the requirements:

```text
python -m pip install -r requirements.txt
```

4. Keep the mini data supplied in `data/mini/`. Extract the separate full data package so its files appear in `data/full/`.

For environment problems, use the Canvas Python Setup and Troubleshooting page:
https://rmit.instructure.com/courses/157398/pages/python-setup-and-troubleshooting

On macOS with Python from python.org, if an HTTPS request reports `CERTIFICATE_VERIFY_FAILED`, open Applications > Python 3.x > Install Certificates.command for your selected Python version, then retry. See the [Python installation guide](https://docs.python.org/3/using/mac.html#installation-steps). B0 uses Whoosh tokenisation and does not need an NLTK data download.

Obtain your API Base URL, API key, model list, and usage information here:
https://rmit-isys1079-3476-gateway-aze5hgbtavg2a2bt.australiaeast-01.azurewebsites.net/course/isys1079-3476

If imports fail, confirm that VS Code opened this folder and run commands with `python -m` from its terminal. See:
https://rmit.instructure.com/courses/157398/pages/python-project-imports-and-running-modules

## Three Workflow Stages

Run these from this folder:

```text
python -m index.build --config mini_config.ini --api_key YOUR_KEY
python -m rag_survey.retrieve --config mini_config.ini --api_key YOUR_KEY
python -m rag_survey.judge --config mini_config.ini --api_key YOUR_KEY
```

> **Start with mini.** Use these mini commands first to confirm that all three stages run and get a rough indication of performance. Move to `full_config.ini` only after the mini workflow is stable. The Main Specification gives the API budget and full-data cost guide.

The starter pipeline writes:

```text
documents -> Whoosh index
questions -> 100 ranked supports -> retrieved.jsonl
retrieved.jsonl -> selected judgments -> judgments.jsonl
ranked support IDs + aggregated judgments -> submission.csv
```

Pass the key only through the documented `--api_key` argument. Do not store an API key in source code or configs.

`call_judging_model(...)` in `judging/judges.py` uses the configured API settings and prints and returns the raw model reply through `api/chat.py`. The cache setting below controls reuse of chat replies. Indexes and saved embedding files are managed separately.

- `[api] use_cache = true`: reuse matching saved replies.
- `use_cache = false`: request fresh replies and update their cache entries.

New replies are saved in both modes, except when the API fails or returns empty or explicitly truncated output. To refresh a problematic reply, rerun judging with `false`. Restore `true` after checking the output. Your method still interprets and checks the reply. Fresh calls use the API budget.

**Content filtering.** The corpus was screened, but the API provider may still reject political or social text by mistake. Retry once. If the error repeats, reduce the batch size to locate the triggering text, then process it separately or adjust the prompt. Rejected replies are not cached, and the final output must still contain `judge_k` judgments.

## B0 and Method Selection

- `mini_config.ini` and `full_config.ini` are your working configs. Both initially select `method = b0`, so the first run works.
- Keep `mini_b0_config.ini` and `full_b0_config.ini` as unchanged B0 comparison configs. The report template provides the full B0 development scores, so a full B0 run and Kaggle submission are optional.
- Your submitted `mini_config.ini` should reproduce the mini version of your final system for the self-check and staff reproducibility check.

To use a method, implement its branch and select it in your working config:

| Module | Branch and file | Config setting |
| --- | --- | --- |
| Retrieval | `bm25` in `retrieval/rankers.py` | Set `method = bm25` in `[retrieval]`. |
| Retrieval | `dense` in `retrieval/rankers.py` | Set `method = dense` in `[retrieval]`. |
| Judging | `llm_judge` in `judging/judges.py` | Set `method = llm_judge` in `[judging]`. |

Select retrieval and judging independently to compare either module with B0. You may add methods or reorganise the internal code while keeping the three commands and final CSV schema working.

**Judging count.** Set `judge_k` in `[judging]` to an integer from 1 to 100. The default is 20. For LLM judging, more documents usually require more input tokens or calls. Actual API use depends on the prompts and cache hits. B0 makes no model calls.

**Tokenisation.** B0 uses the same Whoosh `StandardAnalyzer` settings for indexing, retrieval, and token matching. Your own method may use NLTK or another permitted tokenizer if you apply it consistently.

**Dense retrieval.** Document embeddings and aligned IDs are supplied in `data/mini/` for mini runs and in the full data package for full runs. The working configs point to these files. Use `api/embeddings.py` to encode your query text with `text-embedding-3-small`. `retrieval/vectors.py` loads batches and calculates cosine scores. Appendix Sections 1.4 and 1.5 explain the text construction and memory-mapped loading.

## Submission Checks

Run both checks from the project root after changing the pipeline and before submission.
Use the supplied mini data with the mini version of your final system selected.

**1. Run the mini workflow.**

```text
python -m test_sanity.check_submission --config mini_config.ini --api_key YOUR_KEY
```

The self-check passes your config path to the three stages without reading its fields or choosing an output directory. All three commands must finish within three minutes.

**2. Check and evaluate the new mini CSV.**

```text
python -m metrics.evaluate --submission runs/mini_best/submission.csv
```

The command uses the starter's default mini output path. If you changed the output location, use the actual path to the CSV just generated by your system. Quote the path if it contains spaces.
The evaluator checks the CSV and prints distribution score and MAP@100 using the supplied `data/mini/mini_groundtruth.json`.
Both commands must finish without errors. Record the mini CSV path in the report template's Submission Check.
Mini scores are for debugging. Submit full-data CSVs to both Kaggle tracks for their scores.

## Creating a Comparison Run

For a comparison run, copy `full_config.ini` to a clearly named file such as `access_var.ini`. Change a method or component for Access-Var and Analysis-Var, or one setting for Config-Var. Give `run_dir` a new folder. Keep unrelated settings unchanged so the comparison remains controlled. Components of a combined method may share one function and be controlled by separate config values. In an INI file, edit values on the right of `=`. Keep the section and existing key names used by the runner.

Video: https://1drv.ms/v/c/763539e587cdff66/IQCtY7FfBUN3TY_YiLj_LdLqATdyiaKGaEUi0d9O-jyR5wA?e=6K4c6R
