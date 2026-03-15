# 🧠 AI Data Quality Engine

An autonomous, intelligence-driven data cleaning and parsing product built with Python and Streamlit. This engine is designed to act like a human analyst—inferring the semantic *meaning* of your data columns before applying safe, context-aware cleaning policies.

Unlike basic data cleaners that blindly impute all nulls or delete all duplicates, this engine aggressively protects identifiers, preserves meaningful missing values (like lifecycle timestamps), and documents exactly why it makes every decision.

---

## 🏗️ System Architecture

The engine is built on a highly modular architecture. Responsibility is cleanly separated into specialized worker engines located in the `/utils` directory:

```text
ai-data-cleaning-agent/
│
├── app.py                      # Main Streamlit Dashboard & UI Orchestrator
├── .streamlit/
│   └── config.toml             # Custom Glassmorphism UI & Upload Limit Settings (500MB)
└── utils/
    ├── loader.py               # Robust File Parsing Engine (CSV, JSON, XLSX, TSV, DATA, TXT)
    ├── profiler.py             # Dataset Profiling Engine (Statistical mapping, Cardinality)
    │
    ├── column_intelligence.py  # Column Intent Strategy (Infers if a column is an ID, Timestamp, Categorical, etc.)
    ├── missing_value_policy.py # Missing-Value Decision Framework (Preserve vs. Flag vs. Impute)
    ├── duplicate_policy.py     # Duplicate Handling Framework (ID-aware deduplication)
    │
    ├── cleaner.py              # Cleaning Action Engine (Safe overrides, text stripping, categorical downcasting)
    ├── anomaly.py              # Anomaly Detection Engine (Isolation Forest tracking via sampling)
    ├── explainability.py       # Transparency Report Engine (Logs modifications and intentional skipped actions)
    └── insights.py             # Visual Analysis & Future LLM Integration Engine
```

### 🧠 Core Decision Frameworks

1. **Identifier Protection (`column_intelligence.py` & `loader.py`)**
   - Automatically detects IDs (e.g., `userId`, `transaction_hash`).
   - Immediately forces these values into `string` types at load to prevent pandas from mangling leading zeros or applying scientific notation to large integer IDs.
   - Absolutely prohibits any downstream engine from imputing nulls or normalizing casing on identifiers.

2. **Missing-Value Reasoning (`missing_value_policy.py`)**
   - **Preserve:** Automatically preserves nulls in lifecycle dates (e.g., `closed_date`) because a null implies the event hasn't happened yet.
   - **Flag:** Missing data exceeding safe thresholds (e.g., > 30%) is flagged for manual user review rather than recklessly guessed.
   - **Impute:** Only imputes numeric values (via median) or simple categories (via mode) when mathematically safe.

3. **Safe Deduplication (`duplicate_policy.py`)**
   - Won't blindly drop identical rows.
   - If the dataset has an `identifier` column, identical rows are guaranteed to be a system error and are safely auto-removed.
   - If no identifiers exist (e.g., repeating sensor readings), exact duplicates are preserved and flagged for human review.

4. **Transparent Explainability (`explainability.py`)**
   - Generates a Human-Readable action log natively visible in the Streamlit UI.
   - Explicitly logs fields that were **intentionally bypassed or preserved** due to high modification risk.

---

## 🚀 Running the Engine Locally

### Requirements
- Python 3.9+
- The `venv` virtual environment with `requirements.txt` installed.

### Start the Application
```bash
# 1. Activate your virtual environment
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Mac/Linux

# 2. Run the Streamlit orchestrator
python -m streamlit run app.py
```
The application will launch on `http://localhost:8501/` with the upgraded sleek neon dark theme.

---

## ⚡ Performance Optimizations
- **Statistical Sampling:** Complex operations on datasets >50,000 rows (e.g., Anomaly Training, deep text profiling) use random statistical sampling to prevent server lockup.
- **Categorical Downcasting:** Recognized label columns are `.astype('category')`, shrinking memory footprints dramatically.
- **Concurrent Inference:** Semantic mapping is handled via Python's `ThreadPoolExecutor` for wide datasets. Vectorized Pandas operations run execution actions securely.
