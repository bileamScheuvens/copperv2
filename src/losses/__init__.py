from .criteria import InfoNCERetrievalLoss, LikelihoodRetrievalLoss, MpcRetrievalLoss


def get_fitting_criterion(query_prob: bool, doc_prob: bool, temperature: float):
    if query_prob and doc_prob:
        return MpcRetrievalLoss(temperature)
    if query_prob and not doc_prob:
        return LikelihoodRetrievalLoss(temperature)
    else:
        return InfoNCERetrievalLoss(temperature)
