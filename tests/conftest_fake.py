import time as _t, os as _o
_off=float(_o.environ.get("FAKE_HOUR_OFFSET","0"))*3600
_r=_t.time
_t.time=lambda:_r()+_off
