from src.data.scidocdata import ExtractionsDatamodule
from src.utils.configs import *

cfg = SciDocDataConfig(16, 0.9, 42, SamplerConfig(SamplerType.RANDOM))
dm = ExtractionsDatamodule(cfg)
dm.setup("fit")

loader = dm.train_dataloader()
batch = next(iter(loader))
breakpoint()
