import pandas as pd


class Data:

    def __init__(self) -> None:
        self._keywords: pd.Series = None  ## Expected to contain a series of keywords in pandas.Series format.
        self._unique_keywords: list[str] = None  ## List of unique keywords extracted from the data.
        self._unique_keywords_groups: dict[str, list[str]] = None  ## Dictionary grouping the unique keywords into groups.
                        ## Structure: key (name or identifier of the group) and value (list of keywords belonging to that group).
        self.author_clusters: dict[str, dict] = {}
        self.removed_keywords: list[str] = []
        self.cluster_quality_score: float = -1
    def all_keys_exist(self, keyword_groups_names: list[str]) -> bool: 
        for key in keyword_groups_names:
            if key not in self._unique_keywords_groups.keys():
                return False
        return True

    def merge_keyword_groups(
        self, keyword_groups_names: list[str], new_group_name: str
    ) -> None:
        merged_keywords = []
        for key in keyword_groups_names:
            merged_keywords.extend(self._unique_keywords_groups[key])  ## Extracts the keywords from each group and adds them to a list.
            del self._unique_keywords_groups[key]  ## Removes the original groups from the dictionary.
        self._unique_keywords_groups[new_group_name] = list(set(merged_keywords))  ## Adds a new group with the provided name, assigning the merged keywords (removing duplicates using set).

    @property
    def keywords(self) -> list[str]:
        return self._keywords

    @keywords.setter
    def keywords(self, keywords: list[str]):
        self._keywords = keywords

    @property
    def unique_keywords(self) -> list[str]:
        return self._unique_keywords

    @unique_keywords.setter
    def unique_keywords(self, unique_keywords: list[str]):
        self._unique_keywords = unique_keywords

    @property
    def unique_keywords_groups(self) -> dict[str, list[str]]:
        return self._unique_keywords_groups

    @unique_keywords_groups.setter
    def unique_keywords_groups(self, unique_keywords_groups: dict[str, list[str]]):
        self._unique_keywords_groups = unique_keywords_groups