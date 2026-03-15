# utils/column_intelligence.py
import pandas as pd
import concurrent.futures

class ColumnIntelligenceEngine:
    """
    Infers the semantic role of each column and evaluates modification risk.
    """
    def __init__(self):
        pass

    def infer_semantics(self, df, profile_report):
        """
        Returns a dictionary mapping columns to their inferred semantics securely and concurrently.
        """
        intelligence = {}
        
        def process_col(col_name):
            col_stats = profile_report["columns"].get(col_name, {})
            if not col_stats:
                return col_name, None
            intent, confidence, risk_score = self._infer_single_column(col_name, col_stats, profile_report)
            return col_name, {
                "semantic_type": intent,
                "confidence": confidence,
                "risk_score": risk_score, 
                "reasoning": f"Inferred based on name heuristics, dtype ({col_stats.get('dtype')}), and uniqueness ({col_stats.get('uniqueness_ratio', 0):.2f})."
            }

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_col = {executor.submit(process_col, col): col for col in profile_report["columns"].keys()}
            for future in concurrent.futures.as_completed(future_to_col):
                col_name, intel = future.result()
                if intel:
                    intelligence[col_name] = intel

        return intelligence

    def _infer_single_column(self, col_name, stats, data_profile):
        import re
        name_lower = str(col_name).lower()
        uniqueness = stats.get("uniqueness_ratio", 0)
        is_numeric = stats.get("is_numeric", False)
        unique_cnt = stats.get("unique_count", 0)
        row_count = data_profile.get("row_count", 1)
        
        # 1. Identifiers (IDs, UUIDs, Hashes) - HIGH RISK
        # Strict pattern match for id endpoints
        id_patterns = [r"id$", r"^id_", r"^uuid", r"^guid", r"key$", r"hash"]
        if any(re.search(pat, name_lower) for pat in id_patterns) or "identifier" in name_lower:
            if uniqueness > 0.4: # Even with dupes, if it's named ID and kinda unique, it's an ID.
                return "identifier", 0.95, 1.0 # 1.0 is absolute maximum risk. Do not touch.
            return "identifier", 0.7, 0.95
            
        # 2. Lifecycle & Standard Timestamps
        date_keywords = ["date", "time", "created", "updated", "deleted", "resolved", "closed", "timestamp", "dob", "completed"]
        if any(kw in name_lower for kw in date_keywords):
            lifecycle_kws = ["deleted", "closed", "resolved", "completed", "cancelled", "approved"]
            if any(lw in name_lower for lw in lifecycle_kws):
                return "lifecycle_timestamp", 0.9, 0.95 # Nulls mean state is pending. Imputing ruins it.
            return "timestamp", 0.85, 0.7
            
        # 3. Categorical Labels (State, Status, Type, Category, Boolean-like)
        cat_keywords = ["status", "state", "type", "category", "label", "gender", "role", "is_", "has_"]
        if any(kw in name_lower for kw in cat_keywords) or (unique_cnt < 20 and row_count > 50 and not is_numeric):
            return "categorical", 0.85, 0.6 
            
        # 4. Geographic Fields
        geo_keywords = ["city", "country", "state", "zip", "postal", "address", "lat", "lon", "location"]
        if any(kw in name_lower for kw in geo_keywords):
            return "geographic", 0.85, 0.6
            
        # 5. Free Text (Descriptions, Notes, Comments)
        text_keywords = ["description", "desc", "note", "comment", "summary", "text", "message", "body"]
        if any(kw in name_lower for kw in text_keywords) or (not is_numeric and stats.get("mean_length", 0) > 40):
            return "free_text", 0.85, 0.95 # High risk text. Do not impute.
            
        # 6. Fallbacks
        if is_numeric:
            if uniqueness > 0.9 and "column_" not in name_lower: 
                # highly unique number might actually be a blind identifier (like an order number without 'id' in name)
                return "identifier", 0.4, 0.85 
            return "numeric_measure", 0.7, 0.4 # Safe to occasionally impute
        else:
            if uniqueness > 0.8 and row_count > 100:
                return "identifier", 0.5, 0.85 # probably a random string identifier
            return "string_value", 0.5, 0.5
