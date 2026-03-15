# utils/missing_value_policy.py
class MissingValuePolicyEngine:
    """
    Nuanced decision framework for how to handle nulls per column intent.
    Focuses on preserving meaning and flag-only actions when risky.
    """
    def generate_policy(self, col_name, intel, stats):
        semantic = intel["semantic_type"]
        missing_pct = stats.get("missing_percentage", 0)
        
        policy = {
            "action": "preserve", # Default safe action
            "method": None,
            "reasoning": []
        }
        
        if missing_pct == 0:
            policy["reasoning"].append("No missing values detected.")
            return policy
            
        # 1. High Risk Fields -> ALWAYS PRESERVE
        if semantic in ["identifier", "free_text"]:
            policy["action"] = "preserve"
            policy["reasoning"].append(f"High-risk field ({semantic}). Nulls must be preserved to prevent data hallucination.")
            return policy
            
        # 2. Lifecycle Dates -> PRESERVE (Null means pending)
        if semantic == "lifecycle_timestamp":
            policy["action"] = "preserve"
            policy["reasoning"].append("Lifecycle timestamps frequently have valid nulls (e.g., ticket not yet closed). Preserving.")
            return policy
            
        # 3. Numeric Imputation
        if semantic == "numeric_measure":
            if missing_pct <= 0.3:
                policy["action"] = "impute"
                policy["method"] = "median"
                policy["reasoning"].append(f"Numeric measure with acceptable missingness ({missing_pct:.1%}). Safe to impute median.")
            else:
                policy["action"] = "flag"
                policy["reasoning"].append(f"Too many missing values ({missing_pct:.1%}) to safely impute without skewing distributions.")
            return policy
            
        # 4. Categorical Imputation
        if semantic in ["categorical", "string_value"]:
            if missing_pct <= 0.3:
                policy["action"] = "impute"
                policy["method"] = "mode" if semantic == "categorical" else "unknown_fill"
                policy["reasoning"].append(f"Acceptable missingness. Safely imputing with {policy['method']}.")
            else:
                policy["action"] = "flag"
                policy["reasoning"].append(f"Too much missing data ({missing_pct:.1%}) to guess safely. Flagged instead of imputed.")
            return policy
            
        # Fallback
        policy["reasoning"].append(f"Defaulting to preserve for {semantic} to maintain safety.")
        return policy
