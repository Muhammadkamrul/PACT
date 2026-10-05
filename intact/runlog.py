"""
intact/runlog.py
================
Run folders, structured logging, and crash-safe checkpointing.

REQUIREMENTS THIS FILE SATISFIES
--------------------------------
1. Every run gets its OWN folder, named <timestamp>_<scenario>_<confighash>.
2. The exact config that produced the run is written INTO that folder.
3. Progress is checkpointed, so an interruption (Ctrl-C, OOM, cluster
   pre-emption) does not lose work -- rerun the same command and it resumes.
4. Detailed logging to file AND console, so any error can be traced.

Layout of a run folder:
    runs/20260814-1930_s1_a1b2c3d4e5/
        config.yaml        exact merged config, including CLI overrides
        run.log            full DEBUG log
        events.jsonl       one JSON object per epoch (streamed, append-only)
        decisions.jsonl    one JSON object per inner-loop decision
        checkpoint.pkl     resumable state (written every N epochs)
        results/           final metrics, tables
        figures/           plots
"""
from __future__ import annotations
import json, logging, pickle, shutil, sys, time, os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class RunDir:
    """Owns one run folder and everything written into it."""

    def __init__(self, root: str, scenario: str, cfg: Dict, cfg_hash: str,
                 resume: bool = True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        # --- Resume logic -------------------------------------------------
        # If a folder with the SAME scenario+hash already exists and has a
        # checkpoint, reuse it.  Identical config => identical experiment =>
        # continuing is correct.  A different config gets a different hash
        # and therefore a different folder, so runs can never be mixed up.
        existing = sorted(self.root.glob(f"*_{scenario}_{cfg_hash}"))
        if resume and existing and any(existing[-1].glob("checkpoint_*.pkl")):
            self.path = existing[-1]
            self.resumed = True
        else:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.path = self.root / f"{stamp}_{scenario}_{cfg_hash}"
            self.path.mkdir(parents=True, exist_ok=True)
            self.resumed = False

        (self.path / "results").mkdir(exist_ok=True)
        (self.path / "figures").mkdir(exist_ok=True)

        # Always (re)write the config so the folder is self-describing.
        (self.path / "config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))

        self._setup_logging()
        self.log = logging.getLogger("intact")
        self.log.info("=" * 78)
        self.log.info("RUN FOLDER : %s", self.path)
        self.log.info("RESUMED    : %s", self.resumed)
        self.log.info("CONFIG HASH: %s", cfg_hash)
        for note in cfg.get("_notes", []):
            self.log.info("CONFIG NOTE: %s", note)
        self.log.info("=" * 78)

        # Append-only event streams.  Line-buffered so a crash loses at most
        # the current line, never the whole file.
        self._events = open(self.path / "events.jsonl", "a", buffering=1)
        self._decisions = open(self.path / "decisions.jsonl", "a", buffering=1)

    # -----------------------------------------------------------------
    def _setup_logging(self) -> None:
        logger = logging.getLogger("intact")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s",
            datefmt="%H:%M:%S")
        fh = logging.FileHandler(self.path / "run.log")
        fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO); ch.setFormatter(fmt)
        logger.addHandler(fh); logger.addHandler(ch)
        logger.propagate = False

    # -----------------------------------------------------------------
    def event(self, obj: Dict[str, Any]) -> None:
        """One record per epoch.  Streamed, so partial runs are analysable."""
        self._events.write(json.dumps(obj, default=float) + "\n")

    def decision(self, obj: Dict[str, Any]) -> None:
        """One record per inner-loop decision (there are many per epoch)."""
        self._decisions.write(json.dumps(obj, default=float) + "\n")

    # -----------------------------------------------------------------
    def save_checkpoint(self, state: Dict[str, Any], tag: str = "main") -> None:
        """
        Atomic checkpoint: write to a temp file, then rename.  A crash
        DURING the write therefore cannot corrupt the previous checkpoint.
        """
        tmp = self.path / f"checkpoint_{tag}.pkl.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.path / f"checkpoint_{tag}.pkl")
        self.log.debug("checkpoint saved at epoch %s", state.get("epoch"))

    def load_checkpoint(self, tag: str = "main") -> Optional[Dict[str, Any]]:
        p = self.path / f"checkpoint_{tag}.pkl"
        if not p.exists():
            return None
        try:
            with open(p, "rb") as f:
                st = pickle.load(f)
            self.log.info("RESUMING from checkpoint at epoch %s", st.get("epoch"))
            return st
        except Exception as e:                      # corrupt checkpoint
            self.log.error("checkpoint unreadable (%s) -- starting fresh", e)
            return None

    def save_json(self, name: str, obj: Any) -> None:
        (self.path / "results" / name).write_text(json.dumps(obj, indent=2, default=float))

    def close(self) -> None:
        self._events.close(); self._decisions.close()
