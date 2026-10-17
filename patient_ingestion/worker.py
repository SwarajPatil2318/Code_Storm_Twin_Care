"""Bounded parser subprocess: sensitive stdout is captured privately by the service."""
import contextlib
import io
import json
from pathlib import Path
import resource
import sys
from .parsers import parse
from .schema import IngestionError


def main():
    resource.setrlimit(resource.RLIMIT_AS,(768*1024*1024,768*1024*1024))
    resource.setrlimit(resource.RLIMIT_CPU,(10,10))
    resource.setrlimit(resource.RLIMIT_FSIZE,(0,0))
    path,format,document_id,filename=sys.argv[1:]
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            result=parse(Path(path).read_bytes(),format,document_id,filename)
        print(json.dumps({'ok':True,'result':result},allow_nan=False))
    except IngestionError as error:
        print(json.dumps({'ok':False,'code':error.code,'message':error.message}))
    except Exception:
        print(json.dumps({'ok':False,'code':'PARSE_FAILED','message':"We couldn't reliably read health information from this file. Try another document or enter it manually."}))


if __name__=='__main__':main()
