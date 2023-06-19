from .pull_seg_train_framework import PullSegmentationMapTrainFramework





class PullSegmentationMapTrainFrameworkWith2dConstraints(PullSegmentationMapTrainFramework):
    def __init__(self, train_loader, valid_loader, model, loss_func, args) -> None:
        super().__init__(train_loader, valid_loader, model, loss_func, args)
    
    def _prepare_data(self, d):
        data = super()._prepare_data(d)
        data["2d_constraints"] = d["2d_constraints"]
        return data



class PullSegmentationMapTrainFrameworkWith2dConstraintsInference(PullSegmentationMapTrainFramework): # TODO
    pass
