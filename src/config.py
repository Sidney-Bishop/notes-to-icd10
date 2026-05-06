"""
config.py - Project-wide configuration singleton for the Surgical Era project.

Implements a declarative, parametric configuration system supporting:
- Medallion Architecture (raw/silver/gold data layers)
- Polars DataFrame I/O with Parquet
- DuckDB connection management with context manager support
- Structured JSONL logging for audit trails
- Automated directory provisioning

Author: Jason Roche
Last Updated: 2026-02-20
"""

import yaml
import json
import os
import polars as pl
import duckdb
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from contextlib import contextmanager


def find_project_root(start: Path = None) -> Path:
    """
    Walk upward from ``start`` (default: cwd) until ``artifacts.yaml`` is found.

    This is the canonical project root resolver used by all scripts and notebooks.
    Defined here in ``src/config.py`` — the project's configuration authority —
    so there is exactly one implementation to update if the sentinel file changes.

    The function intentionally does not import anything from ``src/`` so it can
    be called before the project root has been added to ``sys.path``.

    Parameters
    ----------
    start : Path, optional
        Directory to begin the upward search. Defaults to ``Path.cwd()``.

    Returns
    -------
    Path
        Resolved absolute path to the project root.

    Raises
    ------
    FileNotFoundError
        If ``artifacts.yaml`` is not found in any ancestor directory.
    """
    current = (start or Path.cwd()).resolve()
    while current != current.parent:
        if (current / "artifacts.yaml").exists():
            return current
        current = current.parent
    raise FileNotFoundError(
        "Could not find 'artifacts.yaml' — run from within the project tree, "
        "or pass start=Path('/path/to/project')."
    )


class ArtifactConfig:
    """
    Project-wide source of truth for the Surgical Era.
    
    Implements a singleton pattern to ensure consistent configuration
    across all modules and notebooks. Supports Polars/DuckDB Medallion
    Architecture with automated path resolution and audit logging.
    
    Attributes:
        config_path (Path): Path to the artifacts.yaml configuration file.
        project_root (Path): Root directory of the project.
        config (Dict): Loaded configuration dictionary from YAML.
        db_path (Path): Path to the DuckDB audit trail database.
        log_path (Path): Path to the JSONL processing log file.
    """
    
    _instance: Optional["ArtifactConfig"] = None
    _initialized: bool = False

    def __new__(cls) -> "ArtifactConfig":
        """Singleton pattern: ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize the configuration singleton.
        
        Args:
            config_path: Optional path to artifacts.yaml. If None, searches
                        upward from CWD for the config file.
                        
        Raises:
            FileNotFoundError: If artifacts.yaml cannot be located.
            yaml.YAMLError: If the config file contains invalid YAML.
        """
        # Prevent re-initialization of singleton.
        # If a config_path is supplied but differs from what was used at first
        # init, raise immediately — silently ignoring it causes subtle bugs in
        # tests and scripts that pass an explicit path.
        if self._initialized:
            if config_path is not None:
                given = Path(config_path).resolve()
                if given != self.config_path.resolve():
                    raise ValueError(
                        f"ArtifactConfig already initialised with "
                        f"'{self.config_path}'; cannot reinitialise with '{given}'. "
                        f"Pass config_path=None to reuse the existing singleton."
                    )
            return

        # Resolve config file location
        if config_path is None:
            self.config_path = self._find_config_file()
        else:
            self.config_path = Path(config_path)
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config file not found: {self.config_path}")

        self.project_root = self.config_path.parent.resolve()
        
        # Load and validate configuration
        self.config = self._load_config()
        self._validate_config_structure()
        
        # Provision required directories
        self._create_directories()
        
        # Initialize architecture-specific paths
        self.db_path = self.resolve_path("data", "db") / "audit_trail.ddb"
        self.log_path = self.resolve_path("outputs", "evaluations") / "processing_log.jsonl"
        
        # Mark as initialized to prevent re-init
        self._initialized = True

    def _find_config_file(self) -> Path:
        """
        Search upward from CWD to locate artifacts.yaml.
        
        Returns:
            Path: Absolute path to the found config file.
            
        Raises:
            FileNotFoundError: If config file is not found in current or parent directories.
        """
        current = Path.cwd()
        while current != current.parent:
            config_candidate = current / "artifacts.yaml"
            if config_candidate.exists():
                return config_candidate
            current = current.parent
        raise FileNotFoundError(
            "Could not find 'artifacts.yaml' in current or parent directories. "
            "Please ensure the config file exists in your project root."
        )

    def _load_config(self) -> Dict[str, Any]:
        """
        Load and parse the YAML configuration file.
        
        Returns:
            Dict: Parsed configuration dictionary.
            
        Raises:
            yaml.YAMLError: If YAML parsing fails.
            IOError: If file cannot be read.
        """
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse {self.config_path}: {e}")
        except IOError as e:
            raise IOError(f"Failed to read {self.config_path}: {e}")

    def _validate_config_structure(self) -> None:
        """
        Validate that required top-level config sections exist.
        
        Raises:
            ValueError: If critical configuration sections are missing.
        """
        required_sections = ["paths", "project"]
        missing = [sec for sec in required_sections if sec not in self.config]
        if missing:
            raise ValueError(
                f"Configuration missing required sections: {missing}. "
                f"Please check {self.config_path}"
            )

    def _create_directories(self) -> None:
        """
        Automatically create all directories defined in config paths.

        Iterates through nested path values in the config['paths'] section
        and ensures each directory exists relative to project_root.
        Only string values that look like directory paths are processed —
        values with known file extensions are skipped.
        """
        _FILE_EXTENSIONS = {
            '.parquet', '.csv', '.json', '.jsonl', '.yaml', '.yml',
            '.py', '.db', '.ddb', '.toml', '.lock', '.md', '.txt',
            '.safetensors', '.bin', '.pt',
        }
        paths_config = self.config.get("paths", {})
        for path_val in self._extract_values(paths_config):
            if Path(path_val).suffix.lower() not in _FILE_EXTENSIONS:
                dir_path = self.project_root / path_val
                dir_path.mkdir(parents=True, exist_ok=True)

    def _extract_values(self, node: Union[Dict[str, Any], List[Any]]) -> List[str]:
        """
        Recursively extract all string values from nested config structure.
        
        Args:
            node: Dictionary or list to traverse.
            
        Returns:
            List[str]: Unique string values found (potential paths).
        """
        out = []
        if isinstance(node, dict):
            for v in node.values():
                out.extend(self._extract_values(v))
        elif isinstance(node, list):
            for item in node:
                out.extend(self._extract_values(item))
        elif isinstance(node, str):
            out.append(node)
        return list(set(out))

    def resolve_path(self, *keys: str) -> Path:
        """
        Resolve a nested path from config relative to project root.
        
        Args:
            *keys: Sequence of keys to traverse in config['paths'].
                  Example: resolve_path('data', 'silver') -> config['paths']['data']['silver']
                  
        Returns:
            Path: Absolute, resolved Path object.
            
        Raises:
            KeyError: If any key in the path sequence doesn't exist in config.
        """
        current = self.config.get("paths", {})
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                key_path = " -> ".join(keys)
                raise KeyError(
                    f"Config path not found: '{key_path}'. "
                    f"Available paths under 'paths': {list(self.config.get('paths', {}).keys())}"
                )
            current = current[key]
        
        if not isinstance(current, str):
            raise ValueError(
                f"Config value at '{'.'.join(keys)}' is not a string path: {type(current)}"
            )
            
        return (self.project_root / current).resolve()

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Safely get a nested value from the config dictionary.
        
        Args:
            *keys: Sequence of keys to traverse.
            default: Value to return if path doesn't exist.
            
        Returns:
            The config value at the specified path, or default if not found.
        """
        current = self.config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    # --- DuckDB Connection Management ---
    
    def get_duckdb_conn(self) -> duckdb.DuckDBPyConnection:
        """
        Create and return a NEW DuckDB connection to the audit trail database.
        
        Important: Each call creates a separate connection. The caller is
        responsible for closing the connection using conn.close() or via
        the duckdb_connection() context manager.
        
        Returns:
            duckdb.DuckDBPyConnection: Active connection to audit_trail.ddb.
        """
        return duckdb.connect(str(self.db_path))

    @contextmanager
    def duckdb_connection(self):
        """
        Context manager for DuckDB connections to ensure proper cleanup.
        
        Usage:
            with config.duckdb_connection() as conn:
                result = conn.execute("SELECT * FROM table").df()
                
        Yields:
            duckdb.DuckDBPyConnection: Active connection (auto-closed on exit).
        """
        conn = self.get_duckdb_conn()
        try:
            yield conn
        finally:
            conn.close()

    def close_duckdb_conn(self) -> None:
        """
        NO-OP placeholder for backward compatibility.
        
        Deprecated: DuckDB connections from get_duckdb_conn() are new
        instances per call and should be closed directly by the caller
        using conn.close() or the duckdb_connection() context manager.
        """
        # Intentionally empty - see docstring
        pass

    # --- Polars DataFrame I/O (Medallion Architecture) ---
    
    def register_dataframe(
        self, 
        name: str, 
        df: pl.DataFrame, 
        layer: str = "silver", 
        phase: str = "EDA",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save a Polars DataFrame to the specified Medallion layer as Parquet.
        
        Args:
            name: Artifact name (filename without extension).
            df: Polars DataFrame to persist.
            layer: Medallion layer ('raw', 'silver', or 'gold'). Default: 'silver'.
            phase: Pipeline phase for logging (e.g., 'EDA', 'training', 'evaluation').
            metadata: Optional additional metadata to include in audit log.
            
        Returns:
            Path: Absolute path to the saved Parquet file.
            
        Raises:
            ValueError: If layer is not one of the valid Medallion layers.
        """
        valid_layers = {"raw", "silver", "gold"}
        if layer not in valid_layers:
            raise ValueError(f"Invalid layer '{layer}'. Must be one of: {valid_layers}")
        
        output_dir = self.resolve_path("data", layer)
        file_path = output_dir / f"{name}.parquet"
        
        # Write using Polars native method (efficient, preserves schema)
        df.write_parquet(file_path, compression="zstd")
        
        # Log the registration event with optional metadata
        log_details = {
            "name": name,
            "path": str(file_path),
            "shape": df.shape,
            "columns": df.columns,
            **(metadata or {})
        }
        
        self.log_event(
            phase=phase,
            action=f"register_{layer}_df",
            details=log_details
        )
        
        print(f"✓ Registered {layer} artifact '{name}' ({df.shape[0]:,} rows, {df.shape[1]} cols)")
        return file_path

    def load_dataframe(self, name: str, layer: str = "silver") -> pl.DataFrame:
        """
        Load a Polars DataFrame from the specified Medallion layer.
        
        Args:
            name: Artifact name (filename without extension).
            layer: Medallion layer ('raw', 'silver', or 'gold'). Default: 'silver'.
            
        Returns:
            pl.DataFrame: Loaded DataFrame.
            
        Raises:
            FileNotFoundError: If the Parquet file does not exist.
        """
        file_path = self.resolve_path("data", layer) / f"{name}.parquet"
        
        if not file_path.exists():
            available = list(file_path.parent.glob("*.parquet"))
            raise FileNotFoundError(
                f"Artifact not found: {file_path}\n"
                f"Available artifacts in '{layer}' layer: {[f.stem for f in available]}"
            )
        
        return pl.read_parquet(file_path)

    # --- Structured Audit Logging ---
    
    def log_event(
        self, 
        phase: str, 
        action: str, 
        details: Dict[str, Any],
        user: Optional[str] = None,
        notebook: Optional[str] = None
    ) -> None:
        """
        Append a structured event to the JSONL audit log.
        
        Args:
            phase: Pipeline phase (e.g., 'EDA', 'preprocessing', 'training').
            action: Short action identifier (e.g., 'load_data', 'register_df').
            details: Dictionary of event-specific metadata.
            user: Optional identifier for the user/process triggering the event.
            notebook: Optional name of the notebook/script executing the action.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "action": action,
            "details": details,
        }
        
        # Add optional context fields if provided
        if user:
            entry["user"] = user
        if notebook:
            entry["notebook"] = notebook
            
        # Ensure log directory exists before writing
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    # --- Validation & Utility Methods ---
    
    def validate_config(self) -> Dict[str, List[str]]:
        """
        Validate that critical paths are accessible and writable.
        
        Returns:
            Dict with 'errors' and 'warnings' lists for reporting.
        """
        results = {"errors": [], "warnings": []}
        
        # Check critical paths
        critical_paths = [
            ("data.raw", ["data", "raw"]),
            ("data.silver", ["data", "silver"]),
            ("data.gold", ["data", "gold"]),
            ("outputs.evaluations", ["outputs", "evaluations"]),
        ]
        
        for label, keys in critical_paths:
            try:
                path = self.resolve_path(*keys)
                if not path.exists():
                    results["warnings"].append(f"Path does not exist (will auto-create): {label} -> {path}")
                elif not os.access(path, os.W_OK):
                    results["errors"].append(f"Path not writable: {label} -> {path}")
            except KeyError as e:
                results["errors"].append(f"Config key missing: {label} - {e}")
        
        # Check database path parent directory
        try:
            if not self.db_path.parent.exists():
                results["warnings"].append(f"DB directory will be created: {self.db_path.parent}")
        except Exception as e:
            results["errors"].append(f"Error validating db_path: {e}")
            
        return results

    def __repr__(self) -> str:
        """Human-readable representation for debugging."""
        return f"ArtifactConfig(project_root='{self.project_root}', config_path='{self.config_path.name}')"


# ============================================================================
# Singleton Instance
# ============================================================================
# Import this object anywhere in your project: `from config import config`
config = ArtifactConfig()