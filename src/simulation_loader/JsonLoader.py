from typing import Tuple
from .SimulationLoader import SimulationLoader
import pandas as pd
import json
from tqdm.notebook import tqdm


class JsonLoader(SimulationLoader):
    """Class to load simulation data from JSON files."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_simulations(self) -> pd.DataFrame:
        """Load all simulations from the JSON file and return them as a pandas DataFrame."""
        df: pd.DataFrame = pd.read_json(self.file_path, lines=True)

        # Parse 'in' and 'out' fields from JSON strings to dicts
        df["in"] = df["in"].apply(
            lambda x: pd.json_normalize(eval(x)) if isinstance(x, str) else x
        )
        df["out"] = df["out"].apply(
            lambda x: pd.json_normalize(eval(x)) if isinstance(x, str) else x
        )
        return df

    def retrieve_simulations(self) -> pd.DataFrame:
        """Load simulations from the JSON file into a pandas DataFrame, parsing and normalizing 'in' and 'out' fields. Reads line by line to avoid memory errors."""

        records = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in tqdm(
                list(f), desc="Loading simulations from JSON", unit="lines"
            ):
                if line.strip():
                    obj = json.loads(line)
                    # Parse 'in' and 'out' fields from JSON strings to dicts
                    obj["in"] = (
                        json.loads(obj["in"])
                        if isinstance(obj["in"], str)
                        else obj["in"]
                    )
                    obj["out"] = (
                        json.loads(obj["out"])
                        if isinstance(obj["out"], str)
                        else obj["out"]
                    )
                    records.append(obj)
        df = pd.DataFrame(records)
        # Normalize 'in' and 'out' columns into top-level columns
        in_df = pd.json_normalize(df["in"].tolist())
        in_df = in_df.add_prefix("in.")
        out_df = pd.json_normalize(df["out"].tolist())
        out_df = out_df.add_prefix("out.")
        df = pd.concat([df.drop(columns=["in", "out"]), in_df, out_df], axis=1)
        df.set_index(df.index.astype(str), inplace=True)
        return df

    def retrieve_simulations_io(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load simulations and their I/O data from the JSON file into pandas DataFrames."""
        import json

        df = pd.read_json(self.file_path, lines=True)
        # Parse 'in' and 'out' fields from JSON strings to dicts
        df["in"] = df["in"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        df["out"] = df["out"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
        # Use row number as index
        df.set_index(df.index.astype(str), inplace=True)
        inputs = df["in"].apply(pd.Series)
        outputs = df["out"].apply(pd.Series)
        inputs.index = df.index
        outputs.index = df.index
        return inputs, outputs

    def __enter__(self) -> "JsonLoader":
        """Establish a connection to the JSON file (if needed)."""
        # No actual connection needed for JSON files
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> "JsonLoader":
        """Disconnect from the JSON file (if needed)."""
        # No actual disconnection needed for JSON files
        return self
