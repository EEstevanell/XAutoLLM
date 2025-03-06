import math
from typing import List
from autogoal.search.utils import crowding_distance, feature_scaling, non_dominated_sort
from ._pge import PESearch


class NSPESearch(PESearch):
    def __init__(
        self,
        *args,
        ranking_fn=None,
        **kwargs,
    ):
        def default_ranking_fn(_, fns):
            rankings = [-math.inf] * len(fns)
            fronts = non_dominated_sort(fns, self._maximize)
            for ranking, front in enumerate(fronts):
                for index in front:
                    rankings[index] = -ranking
            return rankings

        if ranking_fn is None:
            ranking_fn = default_ranking_fn

        super().__init__(
            *args,
            ranking_fn=ranking_fn,
            **kwargs,
        )

    def _indices_of_fittest(self, fns: List[List[float]]):
        fronts = non_dominated_sort(fns, self._maximize)
        indices = []
        k = int(self._selection * len(fns))

        for front in fronts:
            if len(indices) + len(front) <= k:
                indices.extend(front)
            else:
                indices.extend(
                    sorted(front, key=lambda i: -crowding_distance(fns, front, self._maximize, i))[
                        : k - len(indices)
                    ]
                )
                break
        return indices