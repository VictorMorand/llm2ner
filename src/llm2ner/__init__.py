import os
# by default, set env to offline mode. Othewise api call slows everything down...
# Will be overwritten if needed to download models or datasets
HF_OFFLINE = "1"  # set to "0" to go online

for var in [
    "HF_EVALUATE_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
]:
    os.environ[var] = HF_OFFLINE

from . import utils
from . import losses
from . import data
from . import masks
from . import models
from . import tasks
from . import results
from . import heuristics
from . import distillation
from . import classification
from . import decoders
from . import baselines
from . import plotting

# Expose key classes and functions
from .models import NERmodel, ToMMeR
