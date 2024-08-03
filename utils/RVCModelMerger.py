import torch
import os
from MergeModels import merge_model
from utils.ModelMerger import ModelMerger, ModelMergerRequest


class RVCModelMerger(ModelMerger):
    @classmethod
    def merge_models(cls, request: ModelMergerRequest):
        merged, success = merge_model(request)

        if success:
            is_exist = os.path.exists("merges")

            if not is_exist:
                os.mkdir("merges")

            file_loc = os.path.join("merges", request.mergedName + ".pth")
            torch.save(merged, file_loc)

        return merged, success
