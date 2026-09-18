"""Explicit input-only parser for the published mixed-batch binary fixtures."""
from dataclasses import dataclass
from pathlib import Path
import hashlib, struct, sys
from .schema import Task

@dataclass(frozen=True)
class FieldParameters:
    meta: int
    land_ip: bytes
    land_fp: bytes
    tree_ip: bytes
    tree_fp: bytes
    sha256: str

@dataclass(frozen=True)
class Geometry:
    task_id: str
    k: int
    n: int
    m: int
    centers: bytes
    endpoints: bytes
    offsets: bytes
    owners: bytes
    def task(self,field,*,trace=False,audit=False):
        return Task(self.task_id,field.binding(self.k,trace=trace,audit=audit),self.n,self.m,
                    self.centers,self.endpoints,self.offsets,self.owners)

def _read(path,expected):
    data=Path(path).read_bytes(); actual=hashlib.sha256(data).hexdigest()
    if expected is not None and actual.lower()!=expected.lower():
        raise ValueError('Input SHA256 mismatch: '+Path(path).name)
    return data,actual

def load_inputs(fields_path,jobs_path,*,fields_sha256=None,jobs_sha256=None):
    """Read only on an explicit call; assets and expected hashes are caller supplied."""
    if sys.byteorder!='little': raise ValueError('The published binary schema is little endian')
    data,field_sha=_read(fields_path,fields_sha256)
    if len(data)<20: raise ValueError('Truncated fields header')
    *sizes,meta=struct.unpack_from('<5i',data)
    if min(sizes)<0 or 20+4*sum(sizes)!=len(data): raise ValueError('Fields length/EOF mismatch')
    at=20; blobs=[]
    for count in sizes:
        blobs.append(data[at:at+count*4]);at+=count*4
    fields=FieldParameters(meta,*blobs,field_sha)
    data,job_sha=_read(jobs_path,jobs_sha256)
    if len(data)<4: raise ValueError('Truncated jobs header')
    count,=struct.unpack_from('<i',data)
    if not 0<=count<=1000: raise ValueError('Invalid job count')
    at=4;jobs=[];indices={}
    for _ in range(count):
        if at+12>len(data): raise ValueError('Truncated job shape')
        k,n,m=struct.unpack_from('<3i',data,at);at+=12
        if not 0<=k<=6 or min(n,m)<0 or (n==0 and m!=0): raise ValueError('Invalid job shape')
        widths=(24*n,24*m,4*(n+1),4*m)
        if at+sum(widths)>len(data): raise ValueError('Truncated geometry')
        arrays=[]
        for width in widths:arrays.append(data[at:at+width]);at+=width
        index=indices.get(k,0);indices[k]=index+1
        jobs.append(Geometry(f'K{k}_batch{index}',k,n,m,*arrays))
    if at!=len(data): raise ValueError('Trailing job bytes')
    return fields,jobs,dict(fields_sha256=field_sha,jobs_sha256=job_sha,schema='mixed-batch-little-endian-v1')