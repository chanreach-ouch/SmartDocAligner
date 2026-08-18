try:
    import turbojpeg
    class DummyTurboJPEG:
        def __init__(self, *args, **kwargs): pass
    turbojpeg.TurboJPEG = DummyTurboJPEG
except ImportError:
    pass

import docaligner
import inspect

print(inspect.getsource(docaligner.DocAligner.__init__))
