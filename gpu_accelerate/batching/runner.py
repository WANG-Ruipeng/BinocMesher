"""Explicit offline grouping of ready tasks. No work is performed on import."""
from dataclasses import dataclass
from .schema import Task, BudgetExceeded
from .pack import pack_tasks, split_outputs, IncompatibleTasks

@dataclass(frozen=True)
class BatchResult:
    outputs: dict
    selected_path: str
    task_ids: tuple
    fallback_reason: str | None = None

class BatchExecutionError(RuntimeError):
    """The original failure and a separate cleanup failure are both retained."""
    def __init__(self, primary, cleanup):
        self.primary=primary; self.cleanup=cleanup
        super().__init__(f'batch failed: {primary!r}; cleanup also failed: {cleanup!r}')

def _check_tasks(field, tasks):
    seen=set()
    for task in tasks:
        if not isinstance(task,Task): raise TypeError('Expected immutable Task objects')
        if task.task_id in seen: raise ValueError('Duplicate task_id: '+task.task_id)
        seen.add(task.task_id)
        key=task.binding
        if key.backend!='published': raise NotImplementedError('OPTIONAL_BACKEND_NOT_PACKAGED: graph')
        if key != field.binding(key.k,trace=key.trace,audit=key.audit):
            raise ValueError('Task belongs to a different or expired Field/device/context/profile')

def _run_created(solver):
    try:
        solver.prepare()
        solver.run()
        outputs=solver.readback()
        errors=solver.validation()
        if any(errors): raise RuntimeError('Nonzero solver validation: '+repr(errors))
        result={name:bytes(value) for name,value in outputs.items()}
        required={'position','witness','valid','left','right'}
        if not required.issubset(result): raise RuntimeError('Missing required five outputs')
    except BaseException as primary:
        try: solver.close()
        except BaseException as cleanup: raise BatchExecutionError(primary,cleanup) from primary
        raise
    solver.close()
    return result

def _serial(field,tasks,path,reason=None):
    # Results remain private until every task and cleanup succeeds.
    result={}
    for task in tasks:
        result[task.task_id]=_run_created(field.create_solver(task))
    return BatchResult(result,path,tuple(task.task_id for task in tasks),reason)

def solve_many(field,tasks,*,batching='off',backend='published',fallback='error',max_pack_bytes=2**31):
    """Solve already-ready tasks; default serial, with no waiting or queue.

    Field construction/upload is caller-owned. The offline path includes packing,
    new solver allocation/upload/prepare, complete reset/solve/readback, owned
    output copies and destruction. This function makes no timing promise.
    Only incompatible grouping or a pre-submit allocation rejection can explicitly
    fall back to serial. Any error after run begins propagates without retry.
    """
    if backend!='published':
        raise NotImplementedError('OPTIONAL_BACKEND_NOT_PACKAGED: '+str(backend))
    if batching not in ('off','offline'): raise ValueError('batching must be off or offline')
    if fallback not in ('error','serial'): raise ValueError('fallback must be error or serial')
    tasks=tuple(tasks)
    if not tasks: return BatchResult({},batching,())
    _check_tasks(field,tasks)
    if batching=='off': return _serial(field,tasks,'off')
    try: packed=pack_tasks(tasks,max_bytes=max_pack_bytes)
    except (IncompatibleTasks,BudgetExceeded) as error:
        if fallback=='serial': return _serial(field,tasks,'serial_fallback',str(error))
        raise
    # Importing the binding module does not load its DSO. Construction is explicit.
    from .backend import PreSubmitBudgetError
    try: solver=field.create_solver(packed.task)
    except PreSubmitBudgetError as error:
        if fallback=='serial': return _serial(field,tasks,'serial_fallback',str(error))
        raise
    values=_run_created(solver)
    result=split_outputs(values,packed.segments)
    if tuple(result)!=tuple(task.task_id for task in tasks):
        raise RuntimeError('Output routing did not preserve task identity/order')
    return BatchResult(result,'offline',tuple(result))