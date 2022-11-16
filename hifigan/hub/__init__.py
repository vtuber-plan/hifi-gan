CKPT_URLS = {
    "hifigan-48k": "https://github.com/vtuber-plan/hifi-gan/releases/download/v0.0.2/hifigan_48k.pt",
}
import torch
from ..model.generators.generator import Generator

def hifigan_48k(
    pretrained: bool = True,
    progress: bool = True,
) -> Generator:
    hifigan = Generator(
        initial_channel=256,
        resblock="1",
        resblock_kernel_sizes=[3,7,11],
        resblock_dilation_sizes=[
            [1,3,5],
            [1,3,5],
            [1,3,5]
        ],
        upsample_rates=[8,8,4,2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16,16,4,4]
        )
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            CKPT_URLS["hifigan-48k"], progress=progress
        )
        hifigan.load_state_dict(checkpoint)
        hifigan.eval()
    return hifigan