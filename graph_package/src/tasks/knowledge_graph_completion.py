from torchdrug.tasks import KnowledgeGraphCompletion as KnowledgeGraphCompletion_TD


class KnowledgeGraphCompletion(KnowledgeGraphCompletion_TD):
    def __init__(
        self,
        model,
        criterion="bce",
        metric=("mr", "mrr", "hits@1", "hits@3", "hits@10"),
        num_negative=128,
        margin=6,
        adversarial_temperature=0,
        strict_negative=True,
        filtered_ranking=True,
        fact_ratio=None,
        sample_weight=True,
    ):
        super().__init__(
            model=model,
            criterion=criterion,
            metric=metric,
            num_negative=num_negative,
            margin=margin,
            adversarial_temperature=adversarial_temperature,
            strict_negative=strict_negative,
            filtered_ranking=filtered_ranking,
            fact_ratio=fact_ratio,
            sample_weight=sample_weight,
        )
