# utils/duplicate_policy.py
class DuplicatePolicyEngine:
    """
    Determines duplicate removal safety based on the presence of ID columns.
    If no ID columns exist, repeated numeric rows might be valid.
    """
    def generate_global_policy(self, intelligence_dict, df_stats):
        dup_ratio = df_stats.get("duplicate_ratio", 0)
        
        policy = {
            "drop_exact_duplicates": False,
            "reasoning": []
        }
        
        if dup_ratio == 0:
            policy["reasoning"].append("No exact duplicates found.")
            return policy
            
        # Does the dataset contain at least one recognized Identifier?
        has_id = any(info["semantic_type"] == "identifier" for info in intelligence_dict.values())
        
        if has_id:
            policy["drop_exact_duplicates"] = True
            policy["reasoning"].append("Dataset contains a unique identifier column. Exact row duplicates are therefore guaranteed to be system errors rather than legitimate repeated observations. Safe to auto-remove.")
        else:
            policy["drop_exact_duplicates"] = False
            policy["reasoning"].append("No unique identifiers detected. These exact duplicate rows could be valid repeated measurements (e.g., sensor readings). Preserving duplicates; flagged for manual review.")
            
        return policy
