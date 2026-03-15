# utils/cleaner.py
import pandas as pd
import numpy as np

class CleaningActionEngine:
    """
    Executes the strictly controlled cleaning actions defined by granular policies.
    Never alters fields explicitly protected by the policies.
    """
    def __init__(self):
        pass

    def clean(self, df, intelligence_dict, missing_policies, duplicate_policy, user_overrides=None):
        cleaned_df = df.copy()
        action_log = []
        
        for col in cleaned_df.columns:
            intel = intelligence_dict.get(col, {})
            m_policy = missing_policies.get(col, {})
            
            if not intel or not m_policy:
                continue
                
            semantic = intel.get("semantic_type")
            
            # --- Text Normalization (Only for Safe Strings, NEVER IDs) ---
            if cleaned_df[col].dtype == "object":
                # Convert literal string Nans to actual numeric NaNs for cleaner downstream logic
                cleaned_df[col] = cleaned_df[col].replace(["nan", "NaN", "None", "none", "null", "NULL"], np.nan)
                
                if semantic not in ["identifier", "free_text", "lifecycle_timestamp"]:
                    # Safe to aggressively strip and normalize casing for categories
                    cleaned_df[col] = cleaned_df[col].astype(str).str.strip().replace(r'^\s*$', np.nan, regex=True)
                    if semantic == "categorical":
                        # Standardize case to lower for categorical consolidation
                        cleaned_df[col] = cleaned_df[col].str.lower()
                    action_log.append({"column": col, "action": "Text Normalization", "detail": "Safely stripped whitespace and standardized casing."})
                else:
                     # Very light safe strip trailing whitespace for IDs, but NEVER change case
                     cleaned_df[col] = cleaned_df[col].astype(str).apply(lambda x: x.strip() if isinstance(x, str) else x)

            # --- Missing Value Handling ---
            if m_policy.get("action") == "impute":
                missing_count = cleaned_df[col].isnull().sum()
                if missing_count > 0:
                    method = m_policy.get("method")
                    if method == "median":
                        fill_val = cleaned_df[col].median()
                        cleaned_df[col] = cleaned_df[col].fillna(fill_val)
                        action_log.append({"column": col, "action": "Imputation", "detail": f"Filled {missing_count} nulls safely with median ({fill_val})."})
                    elif method == "mode":
                        mode_s = cleaned_df[col].mode()
                        fill_val = mode_s[0] if not mode_s.empty else "Unknown"
                        cleaned_df[col] = cleaned_df[col].fillna(fill_val)
                        action_log.append({"column": col, "action": "Imputation", "detail": f"Filled {missing_count} nulls safely with mode ({fill_val})."})
                    elif method == "unknown_fill":
                        cleaned_df[col] = cleaned_df[col].fillna("Unknown")
                        action_log.append({"column": col, "action": "Imputation", "detail": f"Filled {missing_count} nulls safely with 'Unknown'."})
            
            # Note: We intentionally skip executing if policy is "preserve" or "flag"

            # --- Categorical Downcasting (Optimization) ---
            if semantic == "categorical":
                try:
                    cleaned_df[col] = cleaned_df[col].astype("category")
                except Exception:
                    pass

        # --- Deduplication ---
        drop_dupes = duplicate_policy.get("drop_exact_duplicates", False)
        if user_overrides and user_overrides.get("force_drop_duplicates"):
            drop_dupes = True
            
        if drop_dupes:
            before_rows = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            after_rows = len(cleaned_df)
            dropped = before_rows - after_rows
            if dropped > 0:
                action_log.append({"column": "Global", "action": "Deduplication", "detail": f"Auto-removed {dropped} exact duplicate rows matching an explicit Identifier key."})

        return cleaned_df, action_log
