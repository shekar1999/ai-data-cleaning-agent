# utils/loader.py
import pandas as pd
import json
import io
import chardet
import re

class DataLoader:
    """
    Engine for robust file parsing across multiple file formats.
    Handles delimiter/header inference and safe encoding resolution.
    Immediately protects identifier-like columns from numeric conversion.
    """
    def __init__(self):
        # A lightweight pre-check for forcing IDs as strings during load
        self.id_patterns = [r"id$", r"^id_", r"^uuid", r"^guid", r"key$", r"hash"]

    def _detect_encoding(self, raw_bytes):
        result = chardet.detect(raw_bytes[:20000])
        return result['encoding'] or 'utf-8'

    def _get_sep(self, choice):
        separators = {
            "Comma (,)": ",",
            "Tab (\\t)": "\t",
            "Pipe (|)": "|",
            "Semicolon (;)": ";",
            "Space": r"\s+"
        }
        return separators.get(choice, None)
        
    def _is_id_column(self, col_name):
        name_lower = str(col_name).lower()
        return any(re.search(pat, name_lower) for pat in self.id_patterns)

    def parse_file(self, file_obj, file_name, manual_delimiter="Auto Detect", has_header=True):
        """
        Parses uploaded file object and returns:
        - df: pandas DataFrame
        - parsing_report: dict containing parse confidence, warnings, detected delimiter, etc.
        """
        report = {
            "parsing_confidence": "High",
            "warnings": [],
            "detected_delimiter": None,
            "encoding": "utf-8"
        }

        file_name_lower = file_name.lower()
        
        # When user checks 'no header', we pass None to pandas so it doesn't eat the first row
        header_option = 0 if has_header else None

        raw_bytes = file_obj.read()
        file_obj.seek(0)

        text_data = ""
        if not file_name_lower.endswith((".xlsx", ".json")):
            encoding = self._detect_encoding(raw_bytes)
            report["encoding"] = encoding
            try:
                text_data = raw_bytes.decode(encoding, errors='replace')
            except Exception as e:
                text_data = raw_bytes.decode('utf-8', errors='ignore')
                report["warnings"].append(f"Encoding issue: {str(e)}. Fallback to utf-8 ignore.")
                report["parsing_confidence"] = "Medium"

        # Pre-scan header row to build a dtype dict for known IDs
        # We only do this if we actually expect a header
        id_dtypes = {}
        if header_option == 0 and text_data:
            try:
                # Peek at the first line to grab col names
                first_line = text_data.split('\n', 1)[0]
                sep = "," if manual_delimiter in ("Auto Detect", "Comma (,)") else self._get_sep(manual_delimiter)
                if not sep and file_name_lower.endswith((".txt", ".data", ".tsv")):
                    # guess separator for peek
                    for s in [",", "\t", "|", ";"]:
                        if s in first_line:
                            sep = s
                            break
                            
                if sep:
                     raw_cols = pd.read_csv(io.StringIO(first_line), sep=sep, nrows=0, engine='python').columns
                     for col in raw_cols:
                         if self._is_id_column(col):
                             id_dtypes[col] = str
            except Exception:
                pass


        try:
            if file_name_lower.endswith(".csv"):
                sep = "," if manual_delimiter in ("Auto Detect", "Comma (,)") else self._get_sep(manual_delimiter)
                report["detected_delimiter"] = sep
                df = pd.read_csv(io.StringIO(text_data), sep=sep, header=header_option, dtype=id_dtypes, engine='python', on_bad_lines='skip')

            elif file_name_lower.endswith(".xlsx"):
                # Pandas read_excel requires string names for dtype dict
                df = pd.read_excel(file_obj, header=header_option, dtype=id_dtypes)

            elif file_name_lower.endswith(".tsv"):
                report["detected_delimiter"] = "\\t"
                df = pd.read_csv(io.StringIO(text_data), sep="\t", header=header_option, dtype=id_dtypes, on_bad_lines='skip')

            elif file_name_lower.endswith((".txt", ".data")):
                sep = self._get_sep(manual_delimiter)
                if sep:
                    report["detected_delimiter"] = sep
                    df = pd.read_csv(io.StringIO(text_data), sep=sep, header=header_option, dtype=id_dtypes, engine='python', on_bad_lines='skip', quotechar='"')
                else:
                    # Robust delimiter sniffing
                    delimiters_to_try = [",", "\t", "|", ";", r"\s+"]
                    df = None
                    valid_dfs = []
                    
                    for trial_sep in delimiters_to_try:
                        try:
                            temp_df = pd.read_csv(
                                io.StringIO(text_data), 
                                sep=trial_sep, 
                                header=header_option, 
                                dtype=id_dtypes,
                                engine='python', 
                                on_bad_lines='skip'
                            )
                            # A good delimiter should produce multiple columns and retain >50% of lines
                            expected_lines = len(text_data.splitlines())
                            if temp_df.shape[1] > 1 and len(temp_df) > (expected_lines * 0.3):
                                valid_dfs.append((trial_sep, temp_df))
                        except Exception:
                            continue

                    if not valid_dfs:
                        raise ValueError("Could not robustly parse TXT/DATA file. Please pick a delimiter manually.")
                        
                    # Pick the delimiter that yielded the most columns & rows securely
                    valid_dfs.sort(key=lambda x: x[1].shape[1] * x[1].shape[0], reverse=True)
                    best_sep, df = valid_dfs[0]
                    report["detected_delimiter"] = best_sep

            elif file_name_lower.endswith(".json"):
                raw_data = json.load(file_obj)
                if isinstance(raw_data, list):
                    df = pd.DataFrame(raw_data)
                elif isinstance(raw_data, dict):
                    df = pd.json_normalize(raw_data)
                else:
                    raise ValueError("Unsupported JSON structure.")
            else:
                raise ValueError("Unsupported file format.")

            # Safe NO-HEADER handling
            if header_option is None and df is not None:
                # User said no headers, so give them safe generic numeric columns
                df.columns = [f"column_{i+1}" for i in range(df.shape[1])]
                
                # We couldn't pre-cast IDs because there was no header name.
                # However, column_intelligence engine will flag high-cardinality cols later and cleaner will protect them.

            if df is not None and df.empty:
                report["warnings"].append("Parsed file returned an empty DataFrame.")
                report["parsing_confidence"] = "Low"

            if id_dtypes:
                report["warnings"].append(f"Protected fields enforced as text: {list(id_dtypes.keys())}")

            return df, report

        except Exception as e:
            report["parsing_confidence"] = "Low"
            report["warnings"].append(f"Parsing failed: {str(e)}")
            return None, report
