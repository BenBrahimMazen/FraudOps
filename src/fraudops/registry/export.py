"""Export the current champion as a self-contained bundle for offline serving.

Downloads the champion pyfunc artifact and writes under the output dir:

    model/       the MLflow model directory (model + pipeline + threshold)
    meta.json    source registry version, model name, export timestamp

The bundle exists so a standalone deployment (the demo image) can serve the
real champion without the tracking server or the object store: a build-time
step re-registers ``model/`` into a local MLflow file store with the
``champion`` alias, and the unmodified serving app loads it from there.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import mlflow

from fraudops.registry import client as registry_client

logger = logging.getLogger("fraudops.registry.export")


def export_champion(out_dir: Path, model_name: str | None = None) -> dict[str, object]:
    """Download the @champion model artifact and its provenance into out_dir."""
    name = model_name or registry_client.model_name()
    client = registry_client.configure()
    version = registry_client.get_version_by_alias(client, registry_client.CHAMPION)
    if version is None:
        raise RuntimeError(f"no @{registry_client.CHAMPION} alias set for {name}")

    dest = out_dir / "model"
    if dest.exists():
        shutil.rmtree(dest)  # fresh export; a stale bundle would serve an old champion
    out_dir.mkdir(parents=True, exist_ok=True)

    uri = f"models:/{name}@{registry_client.CHAMPION}"
    logger.info("downloading %s (version %s) -> %s", uri, version, dest)
    # download to container-local scratch: out_dir may be a bind mount, where
    # renames across the mount boundary fail (EXDEV), and MLflow writes models:/
    # artifacts straight into dst_path rather than a named subdirectory
    scratch = Path(tempfile.mkdtemp())
    try:
        mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=str(scratch))
        model_dir = next(p.parent for p in scratch.rglob("MLmodel"))
        shutil.copytree(model_dir, dest)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    meta: dict[str, object] = {
        "model_name": name,
        "source_version": int(version),
        "exported_at": datetime.now(UTC).isoformat(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export the champion model bundle.")
    parser.add_argument("--out", default="demo", help="output directory (default: demo)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    meta = export_champion(Path(args.out))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
