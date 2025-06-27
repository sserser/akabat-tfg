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
        csvs: list[pd.DataFrame] = self._paper_loader.import_csvs(
            folder_path=folder_path,
            import_column_mappings_by_file=self._preferences.csv_import_column_mappings_by_file,
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

    def generate_single_plot(self, plot_name: str, threshold=0, top_n=None, min_freq=None, filters=None) -> go.Figure:
        project_folder = self._preferences.output_files_folder
        plots_folder = os.path.join(project_folder, self._preferences.plot_folder)
        os.makedirs(plots_folder, exist_ok=True)
        raw_df = self._raw_papers
        author_kw_map = self._cluster_authors.build_author_keyword_map(raw_df)
        author_clusters = self._data.author_clusters
        with open(os.path.join(project_folder, "keyword_groups.json"), "r", encoding="utf-8") as f:
            groups_named = json.load(f)
        filters = filters or {}
        # === Global filters ===
        if filters.get("years"):
            raw_df = raw_df[raw_df["publication_year"].astype(str).isin(filters["years"])]

        if filters.get("keywords"):
            raw_df = raw_df[raw_df["keywords"].apply(lambda kw: any(k in kw for k in filters["keywords"]) if isinstance(kw, str) else False)]

        if filters.get("keyword_groups"):
            group_dict = self._data.unique_keywords_groups  
            all_keywords_from_groups = set(
                kw for g in filters["keyword_groups"] for kw in group_dict.get(g, [])
            )
            raw_df = raw_df[raw_df["keywords"].apply(lambda kw: any(k in kw for k in all_keywords_from_groups) if isinstance(kw, str) else False)]

        if filters.get("authors"):
            raw_df = raw_df[raw_df["author"].apply(lambda a: any(auth in a for auth in filters["authors"]) if isinstance(a, str) else False)]

        if filters.get("author_groups"):
            author_group_dict = self._data.author_clusters  
            all_authors_from_groups = set(
                a for g in filters["author_groups"] for a in author_group_dict.get(g, {}).get("authors", [])
            )
            raw_df = raw_df[raw_df["author"].apply(lambda a: any(auth in a for auth in all_authors_from_groups) if isinstance(a, str) else False)]

        author_citations_map = defaultdict(int)
        for row in raw_df.itertuples(index=False):
            authors_str = getattr(row, "author", "") or getattr(row, "Author", "")
            citations = getattr(row, "citations", 0)
            authors = [a.strip() for a in authors_str.split(";") if a.strip()]
            for author in authors:
                author_citations_map[author] += citations
        df_all = self._db_handler.query_count_unique_papers_per_group_per_year()
        selected_categories = [str(x) for x in df_all["name"].unique()]
        selected_years = [col.split("_")[-1] for col in df_all.columns if col.startswith("paper_count_")]

        df = self._plot_generator.get_filtered_melted_df(
            controller=self,
            category_column_name="Category",
            selected_category_names=selected_categories,
            discard_years=[y for y in df_all.columns if y.endswith(tuple(set(selected_years) ^ set(all_years := [col.split("_")[-1] for col in df_all.columns if col.startswith("paper_count_")])))],
        )

        match plot_name:
            case "polar":
                fig = self._plot_generator.plot_polar(df, "Category", plots_folder, "polar.pdf", groups_named)

                layout_dict = fig.layout.to_plotly_json()
                layout_dict["showlegend"] = filters.get("showlegend", True)

                return {
                    "data": [trace.to_plotly_json() for trace in fig.data],
                    "layout": layout_dict,
                    "frames": [],
                    "config": {"responsive": True}
                }

            case "lines":
                fig = self._plot_generator.plot_lines(df, "Category", plots_folder, "lines.pdf")

                layout_dict = fig.layout.to_plotly_json()
                layout_dict["showlegend"] = filters.get("showlegend", True)

                return {
                    "data": [trace.to_plotly_json() for trace in fig.data],
                    "layout": layout_dict,
                    "frames": [], 
                    "config": {"responsive": True}
                }

            case "log":
                fig = self._plot_generator.plot_log_lines(df, "Category", plots_folder, "growth.pdf")

                layout_dict = fig.layout.to_plotly_json()
                layout_dict["showlegend"] = filters.get("showlegend", True)

                return {
                    "data": [trace.to_plotly_json() for trace in fig.data],
                    "layout": layout_dict,
                    "frames": [],
                    "config": {"responsive": True}
                }

            case "map":
                fig = self._plot_generator.generate_interactive_country_map(
                    raw_df, os.path.join(plots_folder, "country_map.pdf")
                )
                layout_dict = fig.layout.to_plotly_json()
                layout_dict["showlegend"] = filters.get("showlegend", True)
                frames = []
                if hasattr(fig, "frames") and fig.frames:
                    try:
                        frames = [f.to_plotly_json() for f in fig.frames]
                    except Exception as e:
                        print("Error al convertir frames:", e)

                return {
                    "data": [trace.to_plotly_json() for trace in fig.data],
                    "layout": layout_dict,
                    "frames": frames,
                    "config": {"responsive": True}
                }


            case "author_cluster":
                fig = self._plot_generator.generate_interactive_author_cluster_graph(
                    author_clusters=author_clusters,
                    author_citations_map=author_citations_map,
                    save_path_html=os.path.join(plots_folder, "author_clusters.pdf"),
                    threshold=threshold
                )

                layout_dict = fig.layout.to_plotly_json()
                layout_dict["showlegend"] = filters.get("showlegend", True)
                print("[DEBUG] Threshold aplicado:", threshold)
                return {
                    "data": [trace.to_plotly_json() for trace in fig.data],
                    "layout": layout_dict,
                    "frames": [],
                    "config": {"responsive": True}
                }

            case "bubble":
                fig = self._plot_generator.generate_top_keywords_bubble_chart(
                    author_clusters=author_clusters,
                    author_kw_map=author_kw_map,
                    top_n=top_n
                )

                layout_dict = fig.layout.to_plotly_json()
                layout_dict["showlegend"] = filters.get("showlegend", True)

                return {
                    "data": [trace.to_plotly_json() for trace in fig.data],
                    "layout": layout_dict,
                    "frames": [],
                    "config": {"responsive": True}
                }


            case "cumulative_publications":
                fig = self._plot_generator.generate_cumulative_country_map(self._raw_papers)

                # Layout with showlegend
                layout_dict = fig.layout.to_plotly_json()
                layout_dict["showlegend"] = filters.get("showlegend", True)
                frames = []
                if hasattr(fig, "frames") and fig.frames:
                    try:
                        frames = [f.to_plotly_json() for f in fig.frames]
                    except Exception as e:
                        print("Error al convertir frames:", e)
                gif_path = os.path.join(plots_folder, "cumulative_publications.gif") #export gif
                self._plot_generator.export_plotly_animation_as_gif(fig, gif_path, fps=3)

                return {
                    "data": [trace.to_plotly_json() for trace in fig.data],
                    "layout": layout_dict,
                    "frames": frames,
                    "config": {"responsive": True}
                }


            case "heatmap":
                fig = self._plot_generator.generate_interactive_author_keyword_heatmap(
                    author_clusters,
                    author_kw_map,
                    save_path_html=os.path.join(plots_folder, "author_keyword_heatmap.pdf"),
                    min_freq=min_freq
                )

                layout_dict = fig.layout.to_plotly_json()
                layout_dict["showlegend"] = filters.get("showlegend", True)

                return {
                    "data": [trace.to_plotly_json() for trace in fig.data],
                    "layout": layout_dict,
                    "frames": [],
                    "config": {"responsive": True}
                }

            case "impact_map":
                fig = self._plot_generator.generate_citation_heatmap_by_country(self._raw_papers)
                layout_dict = fig.layout.to_plotly_json()
                layout_dict["showlegend"] = filters.get("showlegend", True)
                frames = []
                if hasattr(fig, "frames") and fig.frames:
                    try:
                        frames = [f.to_plotly_json() for f in fig.frames]
                    except Exception as e:
                        print("Error al convertir frames (impact_map):", e)
                gif_path = os.path.join(plots_folder, "citation_map.gif")
                self._plot_generator.export_plotly_animation_as_gif(fig, gif_path, fps=3)
                return {
                    "data": [trace.to_plotly_json() for trace in fig.data],
                    "layout": layout_dict,
                    "frames": frames,
                    "config": {"responsive": True}
                }

            case "cumulative_citations":
                fig = self._plot_generator.generate_cumulative_citation_map_by_country(self._raw_papers)
                layout_dict = fig.layout.to_plotly_json()
                layout_dict["showlegend"] = filters.get("showlegend", True)
                frames = []
                if hasattr(fig, "frames") and fig.frames:
                    try:
                        frames = [f.to_plotly_json() for f in fig.frames]
                    except Exception as e:
                        print("Error al convertir frames (cumulative_citations):", e)
                gif_path = os.path.join(plots_folder, "cumulative_citations.gif")
                self._plot_generator.export_plotly_animation_as_gif(fig, gif_path, fps=3)
                return {
                    "data": [trace.to_plotly_json() for trace in fig.data],
                    "layout": layout_dict,
                    "frames": frames,
                    "config": {"responsive": True}
                }

            case "coauthors":
                fig = self._plot_generator.generate_interactive_coauthorship_graph(
                    raw_df,
                    os.path.join(plots_folder, "coauthors.pdf"),
                    threshold=threshold
                )
                layout_dict = fig.layout.to_plotly_json()
                layout_dict["showlegend"] = filters.get("showlegend", True)
                return {
                    "data": [trace.to_plotly_json() for trace in fig.data],
                    "layout": layout_dict,
                    "frames": [],
                    "config": {"responsive": True}
                }

            case "polar_comparison":
                fig = self._plot_generator.generate_polar_comparison_graph_split_years(
                    raw_df,
                    db_handler=self._db_handler,
                    cluster_keywords=groups_named
                )
                layout_dict = fig.layout.to_plotly_json()
                layout_dict["showlegend"] = filters.get("showlegend", True)
                return {
                    "data": [trace.to_plotly_json() for trace in fig.data],
                    "layout": layout_dict,
                    "frames": [],
                    "config": {"responsive": True}
                }

            case _:
                raise ValueError(f"Unknown plot name: {plot_name}")
