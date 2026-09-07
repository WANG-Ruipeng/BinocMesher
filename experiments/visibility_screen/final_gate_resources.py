"""Final authorized cache ceilings; frozen verification and cleanup are reused.

Only Cave and Forest B are eligible. Geometry, query and report budgets retain
their original preregistered values. This module never creates an amendment.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile

from screen_contracts import read, file_sha, require, load_protocol
from screen_resources import verify_bindings
from forest_native_campaign import PrivateCache, cache_inventory, content_inventory
from final_gate_amendment import load_amendment


def validated_budget(amendment_path, protocol_path, seal_path, segment_id):
    require(segment_id in ('cave', 'forest_b'), 'ONLY_TWO_FIXED_RETRIES_AUTHORIZED')
    receipt = load_amendment(amendment_path, protocol_path, seal_path, segment_id)
    limits = receipt['effective_budgets']
    original = load_protocol(protocol_path)['budgets']
    expected = {**original, 'cache_per_scene_output_bytes': 8*1024**3,
                'cache_per_scene_wall_seconds': 5400}
    require(limits == expected, 'ONLY_FINAL_CACHE_AND_BUILD_WALL_CEILINGS_MAY_CHANGE')
    return receipt


def amendment_binding(path, receipt=None):
    """Common binding shared with the budget-only supervisor, not its worker."""
    return {'path': str(Path(path).resolve()), 'sha256': file_sha(path)}


def remaining_seconds(receipt, now=None):
    deadline = datetime.fromisoformat(receipt['campaign_deadline_utc'])
    require(deadline.tzinfo is not None, 'CAMPAIGN_DEADLINE_REQUIRES_TIMEZONE')
    return max(0.0, (deadline-(now or datetime.now(timezone.utc))).total_seconds())


def require_remaining(receipt):
    remaining = remaining_seconds(receipt)
    require(remaining > 0, 'ORIGINAL_CAMPAIGN_DEADLINE_REACHED')
    return remaining


def amendment_inputs(path, receipt, protocol_path, seal_path):
    bindings = dict(receipt['input_sha256'])
    for item in (path, protocol_path, seal_path):
        bindings[str(Path(item).resolve())] = file_sha(item)
    return bindings


def validate_build_budget(build_report, amendment_path, receipt, protocol_path, segment_id):
    """Bind the new supervisor envelope; never claim its old worker read it."""
    folder = Path(build_report).resolve()
    summary = read(folder/'summary.json')
    contract = read(folder/'build_protocol.json')
    binding = amendment_binding(amendment_path)
    require(summary['status'] == 'COMPLETE_PRE_DISPLACEMENT_OPAQUE_CACHE'
            and summary['segment_id'] == segment_id, 'NEW_AMENDED_BUILD_INCOMPLETE')
    require(summary.get('budget_amendment') == binding
            and contract.get('budget_amendment') == binding, 'BUILD_BUDGET_AMENDMENT_MISMATCH')
    require(file_sha(folder/'build_protocol.json') == summary['build_protocol_sha256'],
            'AMENDED_BUILD_PROTOCOL_CHANGED')
    require(contract.get('effective_budgets') == receipt['effective_budgets'],
            'BUILD_EFFECTIVE_BUDGETS_MISMATCH')
    _, selected = load_protocol(protocol_path, segment_id)
    require(contract['segment'] == selected, 'AMENDED_BUILD_SEGMENT_CHANGED')
    require(contract.get('worker_uses_original_protocol_unchanged') is True
            and contract.get('only_supervisor_native_wall_and_output_ceiling_amended') is True,
            'BUDGET_ONLY_NATIVE_WORKER_CONTRACT_REQUIRED')
    bindings = {str(folder/name): file_sha(folder/name)
                for name in ('summary.json', 'build_protocol.json')}
    bindings.update(summary['input_sha256'])
    verify_bindings(bindings)
    return bindings


class PrivateSceneCache(PrivateCache):
    """The original copy/verify/remove logic, with an authenticated 8 GiB cap."""
    def __init__(self, original, *, budget_amendment, protocol, seal, segment_id,
                 parent='/home/warpwang/binoc-runs'):
        receipt = validated_budget(budget_amendment, protocol, seal, segment_id)
        require_remaining(receipt)
        self.budget_amendment = amendment_binding(budget_amendment)
        self.amendment_inputs = amendment_inputs(budget_amendment, receipt, protocol, seal)
        self.original = Path(original).resolve()
        self.before = cache_inventory(self.original)
        require(sum(row['bytes'] for row in self.before.values())
                <= receipt['effective_budgets']['cache_per_scene_output_bytes'],
                'FINAL_AUTHORIZED_CACHE_COPY_EXCEEDS_8_GIB')
        self.parent = Path(parent).resolve()
        require(self.parent.is_dir(), 'PRIVATE_CACHE_PARENT_MISSING')
        self.temporary = Path(tempfile.mkdtemp(prefix='forest-native-20260906-', dir=self.parent)).resolve()
        self.path = self.temporary/'cache'
        self.closed = False
        require(self.temporary.parent == self.parent, 'PRIVATE_TARGET_OUTSIDE_DECLARED_PARENT')
        try:
            shutil.copytree(self.original, self.path)
            self.copy_before = cache_inventory(self.path)
            require(content_inventory(self.before) == content_inventory(self.copy_before),
                    'PRIVATE_CACHE_CONTENT_DIFFERS')
            verify_bindings(self.amendment_inputs)
            require_remaining(receipt)
        except BaseException:
            self.remove()
            raise
