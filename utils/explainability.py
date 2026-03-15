# utils/explainability.py
import pandas as pd

class ExplainabilityEngine:
    """
    Constructs a highly transparent report detailing not just what was cleaned,
    but importantly what was preserved for human-like safety reasons.
    """
    def __init__(self):
        pass

    def generate_report(self, original_profile, final_profile, intel_dict, missing_policies, dup_policy, action_log):
        report = {
            "summary": {
                "rows_before": original_profile["row_count"],
                "rows_after": final_profile["row_count"],
                "columns": original_profile["column_count"],
                "total_missing_before": original_profile["total_missing"],
                "total_missing_after": final_profile["total_missing"]
            },
            "column_details": [],
            "action_log": action_log,
            "global_policy": dup_policy
        }
        
        for col, col_intel in intel_dict.items():
            m_policy = missing_policies.get(col, {})
            
            # Aggregate reasoning from both engines
            combined_reasoning = []
            combined_reasoning.append(f"INFERENCE: {col_intel.get('reasoning')}")
            if m_policy.get("reasoning"):
                for r in m_policy["reasoning"]:
                    combined_reasoning.append(f"MISSING VALUE SAFETY: {r}")

            detail = {
                "column": col,
                "inferred_semantic": col_intel["semantic_type"],
                "confidence": f"{col_intel['confidence']*100:.0f}%",
                "risk_level": "CRITICAL" if col_intel["risk_score"] >= 0.9 else ("Medium" if col_intel["risk_score"] > 0.4 else "Low"),
                "missing_action": m_policy.get("action", "unknown").upper(),
                "reasoning": combined_reasoning
            }
            report["column_details"].append(detail)
            
        return report
