from .load_model import load_model

models_dict = {
    "setup04/1820500": "https://janelia-cosem-networks.s3.amazonaws.com/v0003.2-pytorch/cosem_models/cosem_models/setup04/1820500",
    "setup04/625000": "https://janelia-cosem-networks.s3.amazonaws.com/v0003.2-pytorch/cosem_models/cosem_models/setup04/625000",
    "setup04/975000": "https://janelia-cosem-networks.s3.amazonaws.com/v0003.2-pytorch/cosem_models/cosem_models/setup04/975000",
    "setup26.1/2580000": "https://janelia-cosem-networks.s3.amazonaws.com/v0003.2-pytorch/cosem_models/cosem_models/setup26.1/2580000",
    "setup26.1/650000": "https://janelia-cosem-networks.s3.amazonaws.com/v0003.2-pytorch/cosem_models/cosem_models/setup26.1/650000",
    "setup28/1440000": "https://janelia-cosem-networks.s3.amazonaws.com/v0003.2-pytorch/cosem_models/cosem_models/setup28/1440000",
    "setup28/775000": "https://janelia-cosem-networks.s3.amazonaws.com/v0003.2-pytorch/cosem_models/cosem_models/setup28/775000",
    "setup36/1100000": "https://janelia-cosem-networks.s3.amazonaws.com/v0003.2-pytorch/cosem_models/cosem_models/setup36/1100000",
    "setup36/500000": "https://janelia-cosem-networks.s3.amazonaws.com/v0003.2-pytorch/cosem_models/cosem_models/setup36/500000",
    "setup45/1634500": "https://janelia-cosem-networks.s3.amazonaws.com/v0003.2-pytorch/cosem_models/cosem_models/setup45/1634500",
    "setup45/625000": "https://janelia-cosem-networks.s3.amazonaws.com/v0003.2-pytorch/cosem_models/cosem_models/setup45/625000",
}

models_list = list(models_dict.keys())

model_names = list(set(x.split("/")[0] for x in models_list))
