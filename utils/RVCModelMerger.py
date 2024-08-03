import torch

from MergeModels import merge_model
from utils.ModelMerger import ModelMerger, ModelMergerRequest


class RVCModelMerger(ModelMerger):
    @classmethod
    def merge_models(cls, request: ModelMergerRequest):
        merged, success = merge_model(request)

        if success:
            torch.save(merged, request.mergedName + ".pth")

        return merged, success
