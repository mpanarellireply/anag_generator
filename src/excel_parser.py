import pandas as pd
from dataclasses import dataclass, field


@dataclass
class RawControlRow:
    """Raw data extracted from one Excel row."""
    controllo: str = ""
    codice_controllo: str = ""
    descrizione_controllo: str = ""
    codice_errore: str = ""
    descrizione_errore: str = ""
    bloccante_warning: str = ""
    messaggio_errore: str = ""
    campo_impattato: str = ""


@dataclass
class RawFunctionData:
    """Raw grouped data for one function from the Excel."""
    function_name: str
    operativita: str = ""
    funzione: str = ""
    parametri_input: str = ""
    rows: list[RawControlRow] = field(default_factory=list)


def read_excel(file_path: str) -> pd.DataFrame:
    """Read the Excel file and return a DataFrame."""
    df = pd.read_excel(file_path)
    return df


def clean_function_name(name: str) -> str:
    """Clean up function names, resolving --> references."""
    if not isinstance(name, str):
        return ""
    name = name.strip()
    if "-->" in name:
        # Take the part after --> as the actual function name
        name = name.split("-->")[-1].strip()
    return name


def group_by_function(df: pd.DataFrame) -> list[RawFunctionData]:
    """Group Excel rows by FUNZIONE.1 (function name) and extract raw data."""
    results = []

    # Filter out rows with no function name
    df_filtered = df[df["FUNZIONE.1"].notna()].copy()
    df_filtered["FUNZIONE.1"] = df_filtered["FUNZIONE.1"].astype(str).str.strip()
    df_filtered = df_filtered[df_filtered["FUNZIONE.1"] != ""]
    df_filtered = df_filtered[df_filtered["FUNZIONE.1"] != " "]

    # Clean function names
    df_filtered["CLEAN_FUNC"] = df_filtered["FUNZIONE.1"].apply(clean_function_name)
    df_filtered = df_filtered[df_filtered["CLEAN_FUNC"] != ""]

    grouped = df_filtered.groupby("CLEAN_FUNC")

    for func_name, group in grouped:
        raw = RawFunctionData(function_name=func_name)

        # Take first non-null values for shared fields
        raw.operativita = _first_non_null(group, "OPERATIVITA")
        raw.funzione = str(_first_non_null(group, "FUNZIONE"))
        raw.parametri_input = _first_non_null(group, "PARAMETRI INPUT")

        # Collect all control rows
        for _, row in group.iterrows():
            control_row = RawControlRow(
                controllo=_safe_str(row.get("CONTROLLO", "")),
                codice_controllo=_safe_str(row.get("CODICE CONTROLLO", "")),
                descrizione_controllo=_safe_str(row.get("DESCRIZIONE CONTROLLO NEW", "")),
                codice_errore=_safe_str(row.get("CODICE ERRORE", "")),
                descrizione_errore=_safe_str(row.get("DESCRIZIONE ERRORE NEW", "")),
                bloccante_warning=_safe_str(row.get("BLOCCANTE/WARNING", "")),
                messaggio_errore=_safe_str(row.get("MESSAGGIO DI ERRORE", "")),
                campo_impattato=_safe_str(row.get("CAMPO IMPATTATO", "")),
            )
            raw.rows.append(control_row)

        results.append(raw)

    return results


def _first_non_null(group: pd.DataFrame, column: str) -> str:
    """Get the first non-null value from a column in a group."""
    if column not in group.columns:
        return ""
    values = group[column].dropna()
    if values.empty:
        return ""
    return str(values.iloc[0]).strip()


def _safe_str(value) -> str:
    """Convert a value to string, handling NaN."""
    if pd.isna(value):
        return ""
    return str(value).strip()
