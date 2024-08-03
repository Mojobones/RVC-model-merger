import torch

from MergeModels import merge_model
from utils.ModelMerger import ModelMerger, ModelMergerRequest


class RVCModelMerger(ModelMerger):
    @classmethod
    def merge_models(cls, request: ModelMergerRequest):
        merged = merge_model(request)

        # いったんは、アップロードフォルダに格納する。（歴史的経緯）
        # 後続のloadmodelを呼び出すことで永続化モデルフォルダに移動させられる。

        torch.save(merged, "merged.pth")