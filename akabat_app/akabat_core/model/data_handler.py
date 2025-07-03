from django.conf import settings
import json
import re
import os
import unicodedata
import numpy as np
from typing import List, Dict, Union, Tuple
from sklearn.preprocessing import normalize
import ast
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter


class UserPreferences:
    def __init__(self):
        """
        Initialize the UserPreferences.
        """
        self.preferences: dict = None
        self.root_folder: str = None
        self.csv_folder: str = None
        self.output_files_folder: str = None
        self.plot_folder: str = None
        self.excluded_starting_by_keywords_at_csv_import: list[str] = []
        self.csv_import_column_mappings_by_file = {}  
        self.csv_separators_by_file = {}
        self.excluded_keywords_at_csv_import: list[str] = []
        self.excluded_contains_keywords_at_csv_import: list[str] = []
        self.excluded_keywords_in_plot: list[str] = []
        self.csv_import_column_names: dict[str, str] = {}
        self.csv_column_names: dict[str, str] = {}


    def save_preferences(self, alternative_preferences_file_path: str = None) -> None:
        preferences_file_path = self.preferences_file_path
        if alternative_preferences_file_path:
            preferences_file_path = alternative_preferences_file_path
        CheckpointHandler.write_to_json_file(self.preferences, preferences_file_path)


class CheckpointHandler:

    @staticmethod
    def write_to_json_file(obj, file_path: str, human_readable: bool = True) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            if human_readable:
                json.dump(obj, f, indent=4)
            else:
                json.dump(obj, f)


    @staticmethod
    def load_from_json_file(file_path: str):
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return False


class PaperLoader:

    def __init__(self) -> None:
        self._transformer_model: SentenceTransformer = None  ## Will be used to obtain embeddings of keywords using SentenceTransformer (sentence-transformers model).
        self._column_names: dict[str, str] = {  ## Dictionary defining column names (for example, "title", "publication_year", "keywords").
            "title": "title",
            "author": "author",
            "publication_year": "publication_year",
            "doi": "doi",
            "country": "country",
            "keywords": "keywords",
            "citations": "citations"
            
        }

        self._column_data_types: dict = { ##Defines expected data type
            "title": str,
            "publication_year": int,
            #"keywords": list[str],
            "author": str,
            "doi": str,
            "country": str,
            "citations": int

        }

    def get_keyword_counts(
        self, df: pd.DataFrame, excluded_keywords: list[str] = None
    ) -> pd.Series:
        if not excluded_keywords:
            excluded_keywords = []

        all_keywords = [ ##Extracts all keywords from each entry
            keyword
            for keywords in df[self._column_names["keywords"]]
            for keyword in keywords
        ]

        keyword_counts = pd.Series(all_keywords).value_counts()

        keyword_counts_filtered = keyword_counts 
        if excluded_keywords: ## Filters discarding keywords selected by users
            keyword_counts_filtered = keyword_counts[
                ~keyword_counts.index.isin(excluded_keywords)
            ]

        return keyword_counts_filtered

    def get_unique_keywords(
        self, df: pd.DataFrame, excluded_keywords: list[str] = None
    ) -> list[str]:
        if not excluded_keywords:
            excluded_keywords = []
        excluded_keywords = [self.normalize_text(k) for k in excluded_keywords]
        all_keywords = self.get_keyword_counts( 
            df=df, excluded_keywords=excluded_keywords
        )
        return [
            keyword
            for keyword in all_keywords.index
            if keyword not in excluded_keywords
        ]
        
    def cluster_centroid_name(self, kw_list: list[str], model, embeddings: np.ndarray) -> str:
        centroid = np.mean(embeddings, axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        return kw_list[np.argmin(distances)]

    def group_keywords_by_semantic_similarity(
        self,
        unique_keywords: list[str],
        k: int = None,
        linkage: str = "ward",
        affinity: str = "euclidean",
        distance_threshold: float = None
    ) -> Union[Dict[str, list[str]], Tuple[Dict[str, list[str]], float]]:

        model_name = "all-mpnet-base-v2"
        if not self._transformer_model:
            self._transformer_model = SentenceTransformer(model_name)

        cleaned_keywords = list(sorted(set(
            self.normalize_text(kw) for kw in unique_keywords if kw.strip()
        )))
        n_samples = len(cleaned_keywords)

        if n_samples < 3:
            raise ValueError("Not enough keywords to form meaningful clusters. At least 3 required.")

        embeddings = self._transformer_model.encode(cleaned_keywords)

        if linkage == "ward":
            affinity = "euclidean"  # only option for ward

        # Silhouette automatic calculation
        if k is None and distance_threshold is None:
            best_score = -1
            best_k = None
            best_labels = None

            max_k = min(n_samples // 2, 40)
            for k_try in range(2, max_k + 1):
                clustering = AgglomerativeClustering(n_clusters=k_try, linkage=linkage, affinity=affinity)
                labels = clustering.fit_predict(embeddings)
                if len(set(labels)) < 2:
                    continue
                try:
                    score = silhouette_score(embeddings, labels)
                except Exception:
                    continue
                if score > best_score:
                    best_score = score
                    best_k = k_try
                    best_labels = labels

            if best_labels is None:
                raise ValueError("Failed to find a valid clustering configuration.")
            labels = best_labels
            silhouette = best_score

        # Manual by K defined
        elif k is not None:
            if k >= n_samples or k < 2:
                raise ValueError(f"Invalid number of clusters: {k}. Must be between 2 and {n_samples - 1}.")
            clustering = AgglomerativeClustering(n_clusters=k, linkage=linkage, affinity=affinity)
            labels = clustering.fit_predict(embeddings)
            silhouette = silhouette_score(embeddings, labels)

        # Distance defined
        else:  # distance_threshold is not None
            clustering = AgglomerativeClustering(
                distance_threshold=distance_threshold,
                n_clusters=None,
                linkage=linkage,
                affinity=affinity
            )
            labels = clustering.fit_predict(embeddings)
            silhouette = -1  # cannot be calculated without k

        # Group keywords
        groups_raw = {}
        for label, keyword in zip(labels, cleaned_keywords):
            groups_raw.setdefault(label, []).append(keyword)

        # Name clusters
        groups_named = {}
        for group_id, kw_list in groups_raw.items():
            emb = self._transformer_model.encode(kw_list)
            rep_name = self.cluster_centroid_name(kw_list, self._transformer_model, emb)
            groups_named[rep_name] = kw_list
        return groups_named, silhouette


    def merge_csvs(self, csvs: list[pd.DataFrame]) -> tuple[pd.DataFrame, int]:
        concatenated_df = pd.concat(csvs)
        return self.remove_duplicates(df=concatenated_df)

    def import_csvs(
        self,
        folder_path: str,
        import_column_mappings_by_file: dict[str, str],
        separators_by_file: dict[str, str],
        keyword_separator: str = ";",
        separator: str = ",",
        header: int = 0,
    ) -> list[pd.DataFrame]:

        csvs = []
        for file_path in os.listdir(folder_path): ##Iterates over all csv files
            if file_path.endswith(".csv"):
                path = os.path.join(folder_path, file_path)
                col_map = import_column_mappings_by_file.get(file_path, {})
                if not col_map:
                    continue
                print(f"Processing file: {file_path}")
                separator = separators_by_file.get(file_path.lower(), ",")

                df = self.import_csv( ##For each csv, calls function import_csv
                        file_path=path,
                        import_columns_names=col_map,
                        keyword_separator=keyword_separator,
                        separator=separator,
                        header=header,
                    )
                if df is not None and not df.empty:
                    df["__source_file__"] = os.path.basename(file_path)
                    csvs.append(df) ##Adds the file to the list of files   
        return csvs
    
    def import_csv(
        self,
        file_path: str,
        import_columns_names: dict[str, str],
        keyword_separator: str = ";",
        separator: str = ",",
        header: int = 0,
    ) -> pd.DataFrame:
        kw_original_col = next(
            (col for col, mapped in import_columns_names.items() if mapped == "keywords"),
            None
        )
        if not kw_original_col:
            raise KeyError("Column 'keywords' not found.")
        expected_cols = list(import_columns_names.keys())
        print(f"FILE: {file_path}")
        print(f"KEY USED: {os.path.basename(file_path).lower()}")
        print(f"SEPARATOR FROM MAP: {separator}")

        try:
            df_raw = pd.read_csv(
                file_path,
                sep=separator,
                header=header,
                encoding="utf-8-sig"
            )
        except pd.errors.ParserError as e:
            raise ValueError(
                f"ParserError while reading '{os.path.basename(file_path)}': {e}. "
                f"Check if the chosen separator ('{separator}') matches the file format."
            )

        missing_cols = [col for col in expected_cols if col not in df_raw.columns]
        if missing_cols:
            raise ValueError(
                f"Error in '{os.path.basename(file_path)}': expected columns {missing_cols} not found. "
                f"Available columns: {list(df_raw.columns)}"
            )
        csv = df_raw[expected_cols].copy()
        if csv.empty or csv.columns.size == 0:
            return
        csv["keywords"] = csv.apply(
            lambda row: self.create_keywords(
                keyword_list=str(row[kw_original_col]),
                keyword_separator=keyword_separator
            ),
            axis=1,
        )
        if kw_original_col != "keywords":
            csv.drop(columns=[kw_original_col], inplace=True)
        inverted_import_columns_names = {
            col: mapped for col, mapped in import_columns_names.items() if mapped != "keywords"
        }
        csv.rename(columns=inverted_import_columns_names, inplace=True)
        csv = csv.astype(self._column_data_types)
        return csv
    
    def normalize_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.lower().strip()


    def apply_keyword_exclusions(
        self,
        df: pd.DataFrame,
        included_keywords: list[str],
        included_starting_by_keywords: list[str],
        included_contains_keywords: list[str],
        excluded_keywords: list[str],
        excluded_starting_by_keywords: list[str],
        excluded_contains_keywords: list[str],
    ) -> pd.DataFrame:
        
        def normalize_list(lst):
            return [self.normalize_text(x) for x in lst if isinstance(x, str)]
        
        included_keywords = normalize_list(included_keywords)
        included_starting_by_keywords = normalize_list(included_starting_by_keywords)
        included_contains_keywords = normalize_list(included_contains_keywords)

        excluded_keywords = normalize_list(excluded_keywords)
        excluded_starting_by_keywords = normalize_list(excluded_starting_by_keywords)
        excluded_contains_keywords = normalize_list(excluded_contains_keywords)

        def filter_keywords(kws):
            if isinstance(kws, str):
                try:
                    kws = ast.literal_eval(kws)
                    if isinstance(kws, str) and ";" in kws:
                        kws = [k.strip() for k in kws.split(";")]
                except Exception:
                    kws = [kws]
            if not isinstance(kws, list):
                kws = []

            kws = [kw for kw in kws if isinstance(kw, str)]

            # Apply whitelist filters
            if included_keywords or included_starting_by_keywords or included_contains_keywords:
                kws = [
                    kw for kw in kws
                    if kw in included_keywords
                    or any(kw.lower().startswith(p.lower()) for p in included_starting_by_keywords)
                    or any(contained.lower() in kw.lower() for contained in included_contains_keywords)
                ]

            # Aply blacklist filters
            kws = [
                kw for kw in kws
                if kw not in excluded_keywords
                and not any(kw.lower().startswith(p.lower()) for p in excluded_starting_by_keywords)
                and not any(contained.lower() in kw.lower() for contained in excluded_contains_keywords)
            ]
            return kws

        df["keywords"] = df["keywords"].apply(filter_keywords)
        return df


    def create_keywords(
        self,
        keyword_list: str,
        keyword_separator: str,
    ) -> list[str]:

        keywords = [
            self.normalize_text(kw.strip())
            for kw in keyword_list.split(keyword_separator)
            if kw.strip()
        ]
        return list(set(keywords))


    def remove_duplicates(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        title_col = self._column_names["title"]
        df["_norm_title"] = df[title_col].apply(self.normalize_text)
        duplicated_num_records = df["_norm_title"].duplicated().sum()
        df_unique = df[~df["_norm_title"].duplicated()].drop(columns=["_norm_title"])
        return df_unique, duplicated_num_records


class AuthorSemanticClustering:

    def __init__(self):
        self.model = SentenceTransformer("all-mpnet-base-v2")

    def build_author_keyword_map(self, df):
        author_map = {}
        for _, row in df.iterrows():
            authors = row.get("author", "")
            keywords = row.get("keywords", [])
            if not isinstance(keywords, list):
                continue
            for author in str(authors).split(";"):
                author = author.strip()
                if author:
                    if author not in author_map:
                        author_map[author] = set()
                    author_map[author].update(keywords)
        return {k: list(v) for k, v in author_map.items()}

    def get_author_embeddings(self, author_keywords):
        return {
            author: self.model.encode(" ".join(keywords))
            for author, keywords in author_keywords.items()
            if keywords
        }

    def cluster_authors_auto(
            self,
            embeddings_dict: dict[str, np.ndarray],
            min_k=5,
            max_k=40,
            linkage="average",
            affinity="euclidean",
            distance_threshold=None
        ):

        authors = list(embeddings_dict.keys())
        embeddings = np.array([embeddings_dict[a] for a in authors])

        # Normalize vectors
        embeddings = normalize(embeddings)

        best_score = -1
        best_k = None
        best_labels = None

        for k in range(min_k, min(max_k + 1, len(embeddings))):
            model = AgglomerativeClustering(
                n_clusters=k,
                linkage=linkage,
                affinity=affinity if linkage != "ward" else "euclidean",
                distance_threshold=distance_threshold
            )
            labels = model.fit_predict(embeddings)
            if len(set(labels)) > 1:
                score = silhouette_score(embeddings, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels

        clusters = {}
        for i, label in enumerate(best_labels):
            clusters.setdefault(label, []).append(authors[i])

        return clusters, best_k, best_score

    
    def evaluate_silhouette_score(self, author_clusters: dict, author_kw_map: dict) -> float:
        all_authors = []
        all_keywords = []

        for cluster_data in author_clusters.values():
            for author in cluster_data["authors"]:
                all_authors.append(author)
                all_keywords.append(author_kw_map.get(author, []))

        if len(set(map(tuple, all_keywords))) < 2:
            return -1 

        mlb = MultiLabelBinarizer()
        X = mlb.fit_transform(all_keywords)
        labels = []
        for cluster_id, data in enumerate(author_clusters.values()):
            labels.extend([cluster_id] * len(data["authors"]))
        try:
            score = silhouette_score(X, labels)
            return score
        except Exception as e:
            return -1

    
    def cluster_authors_by_keywords(
        self,
        papers_df,
        stopword_threshold=0.5,
        top_n_keywords=3,
        k=None,
        distance_threshold=None,
        linkage="average",
        affinity="euclidean"
    ):

        if papers_df.empty:
            raise ValueError("No papers data available for clustering.")

        author_kw_map = self.build_author_keyword_map(papers_df)
        embeddings_dict = self.get_author_embeddings(author_kw_map)

        authors = list(embeddings_dict.keys())
        embeddings = np.array([embeddings_dict[a] for a in authors])
        embeddings = normalize(embeddings)

        if k is None:
            #Automatic mode with Silhouette score
            best_score = -1
            best_k = None
            best_labels = None

            for test_k in range(5, min(31, len(embeddings))):
                model = AgglomerativeClustering(
                    n_clusters=test_k,
                    linkage=linkage,
                    affinity=affinity if linkage != "ward" else "euclidean"
                )
                labels = model.fit_predict(embeddings)
                if len(set(labels)) > 1:
                    score = silhouette_score(embeddings, labels)
                    if score > best_score:
                        best_score = score
                        best_k = test_k
                        best_labels = labels
            labels = best_labels
        else:
            # Manual mode
            model = AgglomerativeClustering(
                n_clusters=k,
                linkage=linkage,
                affinity=affinity if linkage != "ward" else "euclidean",
                distance_threshold=distance_threshold
            )
            labels = model.fit_predict(embeddings)
            best_score = silhouette_score(embeddings, labels) if len(set(labels)) > 1 else -1

        # Group authors
        raw_clusters = {}
        for i, label in enumerate(labels):
            raw_clusters.setdefault(label, []).append(authors[i])

        # Discard stopwords
        total_authors = len(author_kw_map)
        keyword_counter = Counter()
        for kws in author_kw_map.values():
            keyword_counter.update(set(kws))
        stopwords = {kw for kw, count in keyword_counter.items() if count / total_authors > stopword_threshold}

        # Obtains keywords representative for each group
        def get_keywords(authors):
            counter = Counter()
            for a in authors:
                kws = author_kw_map.get(a, [])
                counter.update(kw for kw in kws if kw not in stopwords)
            return [kw for kw, _ in counter.most_common(top_n_keywords)]

        # Name each cluster
        clusters_named = {}
        for cid, authors in raw_clusters.items():
            rep_keywords = get_keywords(authors)
            label = rep_keywords[0] if rep_keywords else f"Group {cid}"
            clusters_named[label] = {
                "authors": authors,
                "keywords": rep_keywords
            }
        return clusters_named, stopwords, best_score
