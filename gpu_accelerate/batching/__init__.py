"""Opt-in offline batching primitives; import does not read inputs or load CUDA."""

from .schema import (BindingKey, BudgetExceeded, DEFAULT_MAX_BYTES, FIVE,
                     IncompatibleTasks, OUTPUTS, OUTPUT_WIDTHS, PackedBatch,
                     Segment, Task, array_bytes, checked_capacity)
from .pack import (compare_outputs, copy_outputs, output_hashes,
                   pack_tasks, split_outputs)

__all__ = ['BindingKey', 'Task', 'Segment', 'PackedBatch', 'IncompatibleTasks',
           'BudgetExceeded', 'DEFAULT_MAX_BYTES', 'FIVE', 'OUTPUTS', 'OUTPUT_WIDTHS',
           'array_bytes', 'checked_capacity', 'pack_tasks', 'split_outputs',
           'copy_outputs', 'compare_outputs', 'output_hashes']
