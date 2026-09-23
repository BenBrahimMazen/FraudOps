"""Registry alias lifecycle tests: register, alias, promote, rollback."""

from __future__ import annotations

import mlflow
import pytest

from fraudops.registry import client as rc
from tests.conftest import MODEL_NAME


def _register_one(registry) -> int:
    """Register another version from a fresh run; returns version."""
    cli = registry["client"]
    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id
    mv = cli.create_model_version(rc.model_name(), source=f"runs:/{run_id}/model", run_id=run_id)
    return int(mv.version)


class TestAliases:
    def test_champion_alias_resolves(self, registry) -> None:
        assert rc.get_version_by_alias(registry["client"], "champion") == str(registry["version"])

    def test_unset_alias_is_none(self, registry) -> None:
        assert rc.get_version_by_alias(registry["client"], "challenger") is None


class TestPromote:
    def test_promote_moves_champion_to_challenger_version(self, registry) -> None:
        new_version = _register_one(registry)
        rc.set_alias(registry["client"], rc.CHALLENGER, str(new_version))
        result = rc.promote_challenger(registry["client"])
        assert result == {"from": str(registry["version"]), "to": str(new_version)}
        assert rc.get_version_by_alias(registry["client"], "champion") == str(new_version)

    def test_promote_without_challenger_raises(self, registry) -> None:
        with pytest.raises(RuntimeError, match="challenger"):
            rc.promote_challenger(registry["client"])


class TestRollback:
    def test_rollback_restores_previous_version(self, registry) -> None:
        new_version = _register_one(registry)
        rc.set_alias(registry["client"], rc.CHAMPION, str(new_version))
        result = rc.rollback_champion(registry["client"])
        assert result["to"] == str(registry["version"])
        assert rc.get_version_by_alias(registry["client"], "champion") == str(registry["version"])

    def test_rollback_to_explicit_version(self, registry) -> None:
        v2 = _register_one(registry)
        v3 = _register_one(registry)
        rc.set_alias(registry["client"], rc.CHAMPION, str(v3))
        result = rc.rollback_champion(registry["client"], to_version=str(v2))
        assert result == {"from": str(v3), "to": str(v2)}


class TestStatus:
    def test_status_reports_aliases_and_versions(self, registry) -> None:
        info = rc.status(registry["client"])
        assert info["model_name"] == MODEL_NAME
        assert info["aliases"]["champion"] == str(registry["version"])
        assert any(v["version"] == registry["version"] for v in info["versions"])
