import pandas as pd
import os 
from akabat_app.akabat_core.model import (
    CheckpointHandler, 
    Data, 
    DBHandler,
    PaperLoader, 
    PlotGenerator, 
    UserPreferences, 
    AuthorSemanticClustering, 
)
import plotly.graph_objects as go
import os
from django.conf import settings
from collections import defaultdict
import json


class Controller:

    def __init__(self, db_path: str = None) -> None:
        self._raw_papers: pd.DataFrame = None
        self._data: Data = Data()
        self._preferences: UserPreferences = UserPreferences()
        if not self._preferences.output_files_folder and db_path:
            self._preferences.output_files_folder = os.path.dirname(db_path)
        if not self._preferences.plot_folder:
            self._preferences.plot_folder = "plots"
        self._paper_loader: PaperLoader = PaperLoader()
        self._cluster_authors: AuthorSemanticClustering = AuthorSemanticClustering()
        if db_path:
            self._db_handler: DBHandler = DBHandler(db_path=db_path)
        else:
            self._db_handler: DBHandler = DBHandler()
        self._plot_generator: PlotGenerator = PlotGenerator()
        self._kill_akabat: bool = False

    def import_all_csvs(self, folder_path: str = None, disabled_indices: set[int] = None) -> int:
        separators_by_file = self._preferences.csv_separators_by_file
        csvs: list[pd.DataFrame] = self._paper_loader.import_csvs(
            folder_path=folder_path,
            import_column_mappings_by_file=self._preferences.csv_import_column_mappings_by_file,
            separators_by_file=separators_by_file,
        )
        merged = pd.concat(csvs, ignore_index=True)
        if self._raw_papers is not None and not self._raw_papers.empty:
            csvs.append(self._raw_papers)

        self._raw_papers, duplicated_number = self._paper_loader.merge_csvs(csvs)
        # Filters in case disabled articles
        if disabled_indices:
            self._raw_papers = self._raw_papers.reset_index(drop=True)
            self._raw_papers = self._raw_papers[~self._raw_papers.index.isin(disabled_indices)]

        return duplicated_number


    def generate_unique_keywords(self) -> None:
        self._data.unique_keywords = self._paper_loader.get_unique_keywords(
            self._raw_papers,
        )

    def save_unique_keywords(self) -> None:
        CheckpointHandler.write_to_json_file(
            self._data.unique_keywords,
            f"{self._preferences.output_files_folder}/unique_keywords.json",
            human_readable=True,
        )

    def load_unique_keywords(self) -> bool:
        loaded_data = CheckpointHandler.load_from_json_file(
            f"{self._preferences.output_files_folder}/unique_keywords.json",
        )
        if loaded_data:
            self._data.unique_keywords = loaded_data
            return True
        return False
    
    def group_keywords_by_semantic_similarity(
        self,
        k: int = None,
        linkage: str = "ward",
        affinity: str = "euclidean",
        distance_threshold=None,
    ) -> tuple[dict, float | None]:
        project_folder = self._preferences.output_files_folder
        groups_json = os.path.join(project_folder, "keyword_groups.json")
        # If groups already created, load them
        if os.path.exists(groups_json):
            with open(groups_json, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if loaded and isinstance(list(loaded.values())[0], dict):
                parsed_groups = {g: d["keywords"] for g, d in loaded.items()}
            else:
                parsed_groups = loaded
            self._data.unique_keywords_groups = parsed_groups
            return parsed_groups, None

        result = self._paper_loader.group_keywords_by_semantic_similarity(
            unique_keywords=self._data.unique_keywords,
            k=k,
            linkage=linkage,
            affinity=affinity,
            distance_threshold=distance_threshold,

        )

        if isinstance(result, tuple): #save results
            groups, silhouette = result
        else:
            groups, silhouette = result, None
        self._data.unique_keywords_groups = groups
        # Save as JSON
        with open(groups_json, "w", encoding="utf-8") as f:
            json.dump(groups, f, indent=2, ensure_ascii=False)
        return groups, silhouette


    def save_keywords_by_semantic_similarity(self) -> None:
        CheckpointHandler.write_to_json_file(
            self._data.unique_keywords_groups,
            f"{self._preferences.output_files_folder}/keyword_groups.json",
            human_readable=True,
        )

    def save_author_clusters(self) -> None:
        CheckpointHandler.write_to_json_file(
            self._data.author_clusters,
            f"{self._preferences.output_files_folder}/author_clusters.json",
            human_readable=True,
        )

    def load_keywords_by_semantic_similarity(self) -> bool:
        loaded_data = CheckpointHandler.load_from_json_file(
            f"{self._preferences.output_files_folder}/keyword_groups.json",
        )
        if loaded_data:
            self._data.unique_keywords_groups = loaded_data
            return True
        return False

    def create_and_populate_database(self) -> None:
        self._db_handler.create_database()
        self._db_handler.populate_paper_table(self._raw_papers)
        self._db_handler.populate_keyword_tables(self._data.unique_keywords_groups)
        self._db_handler.populate_paper_keyword_table(self._raw_papers)
        author_cluster_map = {
            author: cluster_id
            for cluster_id, data in self._data.author_clusters.items()
            for author in data["authors"]
        }
        self._db_handler.populate_paper_author_table(
            df=self._raw_papers,
            author_clusters=author_cluster_map
        )

    def exclude_keyword(self) -> None:
        excluded_keyword = self._viewer.ask_keyword()
        if (
            excluded_keyword
            and excluded_keyword not in self._preferences.excluded_keywords_in_plot
        ):
            self._preferences.excluded_keywords_in_plot.append(excluded_keyword)
            self._preferences.excluded_keywords_in_plot.sort()

    def remove_excluded_keyword(self) -> None:
        excluded_keyword = self._viewer.ask_keyword()
        if excluded_keyword in self._preferences.excluded_keywords_in_plot:
            self._preferences.excluded_keywords_in_plot.remove(excluded_keyword)

    def delete_database(self) -> bool:
        return self._db_handler.delete_database()

    def is_database_created(self) -> bool:
        return self._db_handler.is_database_created()

    def save_preferences(self) -> bool:
        self._preferences.save_preferences("test.json")

    
    def cluster_authors_by_keywords(
        self,
        k: int = None,
        stopword_threshold: float = 0.5,
        top_n_keywords: int = 3,
        distance_threshold=None,
        linkage="average",
        affinity="euclidean"
    ) -> tuple:
        """
        Calls the author clustering module with adjustable parameters.
        If k is None, the automatic mode using Silhouette Score is applied.
        """
        self._data.author_clusters, self._data.removed_keywords, self._data.cluster_quality_score = (
            self._cluster_authors.cluster_authors_by_keywords(
                papers_df=self._raw_papers,
                stopword_threshold=stopword_threshold,
                top_n_keywords=top_n_keywords,
                k=k,
                distance_threshold=distance_threshold,
                linkage=linkage,
                affinity=affinity
            )
        )
        return self._data.author_clusters, self._data.removed_keywords, self._data.cluster_quality_score


    def count_unique_papers_per_group_per_year_from_raw(self, raw_df):

        df = raw_df.copy()
        if "paper_id" not in df.columns:
            df["paper_id"] = df.reset_index().index.astype(str)

        group_dict = self._data.unique_keywords_groups

        def map_to_group(keywords):
            if not isinstance(keywords, list):
                return None
            for group, group_data in group_dict.items():
                if not group_data.get("enabled", True):
                    continue
                kws = group_data.get("keywords", [])
                if any(
                    any(group_kw.lower() in kw.lower() or kw.lower() in group_kw.lower() 
                        for kw in keywords) 
                    for group_kw in kws
                ):
                    return group
            return None


        df["Category"] = df["keywords"].apply(map_to_group)
        print(df["Category"].value_counts(dropna=False))

        if df["Category"].isna().all():
            raise ValueError("No data available after mapping to groups. Revisa los nombres y la lógica de coincidencia.")
        df = df.dropna(subset=["Category"])

        if df.empty:
            raise ValueError("No data available after mapping to groups.")

        df = df[["Category", "publication_year", "paper_id"]].drop_duplicates()

        grouped = df.groupby(["Category", "publication_year"])["paper_id"].nunique().reset_index(name="count")

        if grouped.empty:
            raise ValueError("No data available after grouping.")

        pivot = grouped.pivot(index="Category", columns="publication_year", values="count").fillna(0).astype(int)
        pivot["total_paper_count"] = pivot.sum(axis=1)
        pivot = pivot.reset_index().rename(columns={"Category": "name"})
        pivot = pivot.sort_values("total_paper_count", ascending=False).reset_index(drop=True)

        return pivot
    
    def load_author_clusters(self) -> bool:
        author_clusters_path = os.path.join(self._preferences.output_files_folder, "author_clusters.json")
        if not os.path.exists(author_clusters_path):
            return False

        with open(author_clusters_path, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)

        enabled_clusters = {
            name: data
            for name, data in loaded_data.items()
            if not data.get("disabled", False)
        }

        self._data.author_clusters = enabled_clusters
        return True



    def get_filtered_raw_papers(self, filters):

        df = self._raw_papers.copy()
        years = filters.get("years")
        if years:
            years = [int(y) for y in years]  
            before = len(df)
            df = df[df["publication_year"].isin(years)]
            after = len(df)


        author_groups = filters.get("author_groups")
        if author_groups:
            group_dict = self._data.author_clusters


            author_to_group = {}
            for group in author_groups:
                authors_in_group = group_dict.get(group, {}).get("authors", [])
                for author in authors_in_group:
                    author_to_group[author] = group

            def assign_group(author_str):
                if not isinstance(author_str, str):
                    return None
                for author in author_str.split(";"):
                    author = author.strip()
                    if author in author_to_group:
                        return author_to_group[author]
                return None

            df["AssignedAuthorGroup"] = df["author"].apply(assign_group)
            before = len(df)
            df = df.dropna(subset=["AssignedAuthorGroup"]).copy()
            after = len(df)


            def keep_only_assigned_author_group(author_str, assigned_group):
                if not isinstance(author_str, str):
                    return ""
                valid_authors = [
                    a.strip() for a in author_str.split(";")
                    if author_to_group.get(a.strip()) == assigned_group
                ]
                return ";".join(valid_authors)

            df["author"] = df.apply(
                lambda row: keep_only_assigned_author_group(row["author"], row["AssignedAuthorGroup"]),
                axis=1
            )


        keywords = filters.get("keywords")
        if keywords:
            df = df[df["keywords"].apply(
                lambda kw_list: any(k in kw_list for k in keywords) if isinstance(kw_list, list) else False
            )]

        keyword_groups = filters.get("keyword_groups")
        if keyword_groups:
            group_dict = self._data.unique_keywords_groups
            selected_keywords = set()
            for g in keyword_groups:
                selected_keywords.update(group_dict.get(g, {}).get("keywords", []))
            
            def paper_has_group_keyword(kw_list):
                if not isinstance(kw_list, list):
                    return False
                return any(kw in selected_keywords for kw in kw_list)

            df = df[df["keywords"].apply(paper_has_group_keyword)]

            def keep_only_selected(kw_list):
                if not isinstance(kw_list, list):
                    return []
                return [kw for kw in kw_list if kw in selected_keywords]
            df["keywords"] = df["keywords"].apply(keep_only_selected)

        return df


    def generate_single_plot(
        self, plot_name: str, threshold=0, top_n=None, min_freq=None,
        filters=None, width=1100, height=650, return_figure=False
    ) -> go.Figure:
        project_folder = self._preferences.output_files_folder
        plots_folder = os.path.join(project_folder, self._preferences.plot_folder)
        os.makedirs(plots_folder, exist_ok=True)


        raw_df = self.get_filtered_raw_papers(filters)

        if raw_df.empty or 'publication_year' not in raw_df.columns or 'keywords' not in raw_df.columns:
            raise ValueError("No data available after applying filters or missing required columns.")

 
        author_kw_map = self._cluster_authors.build_author_keyword_map(raw_df)
        author_clusters = self._data.author_clusters
        with open(os.path.join(project_folder, "keyword_groups.json"), "r", encoding="utf-8") as f:
            groups_named = json.load(f)
        self._data.unique_keywords_groups = groups_named
        filters = filters or {}

        author_citations_map = defaultdict(int)
        for row in raw_df.itertuples(index=False):
            authors_str = getattr(row, "author", "") or getattr(row, "Author", "")
            citations = getattr(row, "citations", 0)
            authors = [a.strip() for a in authors_str.split(";") if a.strip()]
            for author in authors:
                author_citations_map[author] += citations

        df_all = self.count_unique_papers_per_group_per_year_from_raw(raw_df)
        df_all = df_all.rename(columns={"name": "Category"})

        selected_categories = df_all["Category"].unique().tolist()
        selected_years = [col for col in df_all.columns if isinstance(col, int)]

 
        df_all.columns = [str(c) for c in df_all.columns]
        if filters.get('years'):
            year_columns = [str(y) for y in filters['years']]
            df_all = df_all[['Category', 'total_paper_count'] + year_columns]


        df = self._plot_generator.get_filtered_melted_df(
            controller=self,
            category_column_name="Category",
            selected_category_names=selected_categories,
            discard_years=[]
        )


        active_authors = set()
        for authors_str in raw_df["author"].dropna():
            for author in authors_str.split(";"):
                author = author.strip()
                if author:
                    active_authors.add(author)

        print(df.head())


        match plot_name:
            case "polar":
                fig = self._plot_generator.plot_polar(
                    df, "Category", plots_folder, "polar.pdf", groups_named,
                    width=width, height=height
                )
            case "lines":
                fig = self._plot_generator.plot_lines(
                    df, "Category", plots_folder, "lines.pdf",
                    width=width, height=height
                )
            case "log":
                fig = self._plot_generator.plot_log_lines(
                    df, "Category", plots_folder, "growth.pdf",
                    width=width, height=height
                )
            case "map":
                fig = self._plot_generator.generate_interactive_country_map(
                    raw_df, os.path.join(plots_folder, "country_map.pdf"),
                    width=width, height=height
                )
            case "author_cluster":
                fig = self._plot_generator.generate_interactive_author_cluster_graph(
                    author_clusters=author_clusters,
                    author_citations_map=author_citations_map,
                    active_authors=active_authors,
                    save_path_html=os.path.join(plots_folder, "author_clusters.pdf"),
                    threshold=threshold,
                    layout=filters.get("layout", "spring"),
                    width=width, height=height
                )
            case "bubble":
                fig = self._plot_generator.generate_top_keywords_bubble_chart(
                    author_clusters=author_clusters,
                    author_kw_map=author_kw_map,
                    top_n=top_n,
                    width=width, height=height
                )
            case "cumulative_publications":
                fig = self._plot_generator.generate_cumulative_country_map(
                    raw_df, width=width, height=height
                )
            case "heatmap":
                fig = self._plot_generator.generate_interactive_author_keyword_heatmap(
                    author_clusters, author_kw_map,
                    save_path_html=os.path.join(plots_folder, "author_keyword_heatmap.pdf"),
                    min_freq=min_freq, width=width, height=height
                )
            case "impact_map":
                fig = self._plot_generator.generate_citation_heatmap_by_country(
                    raw_df, width=width, height=height
                )
            case "cumulative_citations":
                fig = self._plot_generator.generate_cumulative_citation_map_by_country(
                    raw_df, width=width, height=height
                )
            case "coauthors":
                fig = self._plot_generator.generate_interactive_coauthorship_graph(
                    raw_df,
                    os.path.join(plots_folder, "coauthors.pdf"),
                    threshold=threshold,
                    layout=filters.get("layout", "spring"),
                    width=width, height=height
                )
            case "polar_comparison":
                fig = self._plot_generator.generate_polar_comparison_graph_split_years(
                    raw_df,
                    db_handler=self._db_handler,
                    cluster_keywords=groups_named,
                    width=width, height=height
                )
            case _:
                raise ValueError(f"Unknown plot name: {plot_name}")

        layout_dict = fig.layout.to_plotly_json()
        layout_dict["showlegend"] = filters.get("showlegend", True)

        if return_figure:
            return fig

        return {
            "data": [trace.to_plotly_json() for trace in fig.data],
            "layout": layout_dict,
            "frames": getattr(fig, "frames", []),
            "config": {"responsive": True}
        }

