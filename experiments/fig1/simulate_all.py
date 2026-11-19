"""Numerical preparation of columns A and B (field_*, merit_*); cached arrays are optional."""
from pathlib import Path
import argparse
import json
import time
import model  # Set BLAS thread defaults before importing other numerical code.
import simulate_geometry


def prepare(cache, recompute=False, systems=('contracting','chaotic')):
    cache=Path(cache); cache.mkdir(parents=True,exist_ok=True)
    path=cache/'validation.json'
    report=json.loads(path.read_text()) if path.exists() else {}
    for system in systems:
        checks=report.setdefault(system,{})
        start=time.monotonic()
        # data/ ships without the large basis_<system>.npz; compute() writes it on every recomputation.
        files=[cache/f'{kind}_{system}.npz' for kind in ['merit','field']]
        if recompute or not all(p.exists() for p in files):
            checks['geometry']=simulate_geometry.compute(system,cache)
        checks['last_preparation_seconds']=round(time.monotonic()-start,2)
        path.write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True,
                        help='folder to write the arrays into (data/ holds the shipped arrays)')
    parser.add_argument('--recompute',action='store_true')
    parser.add_argument('--system',choices=['both','contracting','chaotic'],default='both')
    args=parser.parse_args()
    systems=['contracting','chaotic'] if args.system=='both' else [args.system]
    prepare(args.cache,args.recompute,systems)
