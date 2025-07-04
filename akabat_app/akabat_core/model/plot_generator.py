import pandas as pd
import plotly.express as px
from pathlib import Path
import numpy as np
import plotly.graph_objs as go
import networkx as nx
from collections import Counter
from typing import Dict, List
import heapq
import plotly.io as pio
import os
import imageio.v2 as imageio
from plotly.subplots import make_subplots
from plotly.graph_objects import Figure


class PlotGenerator:

    @staticmethod
    def get_color_palette(df, category_column_name):
        unique_categories = df[category_column_name].unique()
        num_categories = len(unique_categories)

        plasma_colors = px.colors.sequential.Plasma_r[2:]
        dark24_colors = px.colors.qualitative.Dark24

        if num_categories <= len(plasma_colors):
            indices = np.linspace(0, len(plasma_colors) - 1, num_categories, dtype=int)
            return [plasma_colors[i] for i in indices]
        elif num_categories <= len(dark24_colors):
            return dark24_colors[:num_categories]

        return px.colors.sample_colorscale("rainbow", [i / num_categories for i in range(num_categories)])

    @staticmethod
    def get_df_all(controller, new_category_column_name: str, filename: str = None) -> pd.DataFrame:
        df_all = controller._db_handler.query_count_unique_papers_per_group_per_year()
        print(df_all, "Df_all")
        df_all.rename(columns=lambda x: x.split("_")[-1] if x.startswith("paper_count_") else (new_category_column_name if x.lower() == "name" else x), inplace=True)
        if filename:
            df_all.to_latex(filename)
        return df_all

    @staticmethod
    def get_filtered_df(controller, category_column_name: str, selected_category_names: list[str], discard_years: list[str] = None) -> pd.DataFrame:
        df = PlotGenerator.get_df_all(controller=controller, new_category_column_name=category_column_name)
        df.sort_values('total_paper_count', ignore_index=True, ascending=False, inplace=True)
        df = df[df[category_column_name].isin(selected_category_names)]
        if discard_years is not None:
            existing_years = [col for col in discard_years if col in df.columns]
            df = df.drop(columns=existing_years)
        return df

    @staticmethod
    def get_filtered_melted_df(controller, category_column_name: str, selected_category_names: list[str], discard_years: list[str] = None) -> pd.DataFrame:
        df = PlotGenerator.get_filtered_df(controller=controller, category_column_name=category_column_name, selected_category_names=selected_category_names, discard_years=discard_years)
        df = df.drop('total_paper_count', axis=1)
        df_melted = pd.melt(df, id_vars=[category_column_name], var_name='year', value_name='count')
        print("[DEBUG] Melted DF shape:", df_melted.shape)
        print("[DEBUG] Melted DF head:\n", df_melted.head())
        return df_melted
    
    @staticmethod
    def plot_polar(
        df: pd.DataFrame,
        category_column_name: str,
        plot_folder: str,
        filename: str = "polar_plot.pdf",
        cluster_keywords: Dict[str, List[str]] = None, width: int = 1100, height: int = 650
    ) -> go.Figure:

        if cluster_keywords:
            df["keywords_in_cluster"] = df[category_column_name].map(
                lambda c: "<br>".join(cluster_keywords.get(c, ["(no keywords found)"]))
            )
        else:
            df["keywords_in_cluster"] = "(no keywords available)"

        df = df[df["count"] > 0].copy() ## years with no articles are not shown
        colors = PlotGenerator.get_color_palette(df, category_column_name)

        fig = px.bar_polar(
            df,
            r="count",
            theta="year",
            color=category_column_name,
            template="seaborn",
            color_discrete_sequence=colors,
            custom_data=["keywords_in_cluster"],
            width=width,
            height=height
        )

        fig.update_traces(
            hovertemplate=(
                "<b>Year:</b> %{theta}<br>"
                "<b>Articles:</b> %{r}<br>"
                "<b>Keywords:</b><br>%{customdata[0]}"
            )
        )
        fig.update_layout(title_text="Yearly Distribution of Articles by Category",)
        out = Path(plot_folder)
        out.mkdir(parents=True, exist_ok=True)
        fig.write_image(out / filename)
        return fig

    @staticmethod
    def plot_polars(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        plot_folder: str,
        category_column_name: str,
        filename: str = "polars_plot.pdf",
        save_image: bool = False,
        cluster_keywords: Dict[str, List[str]] = None,
        title1: str = "Left Polar",
        title2: str = "Right Polar",
        width: int = 1100, height: int = 650
    ) -> go.Figure:
        """
        Double polar plot to compare two different periods of time
        """
        def enrich(df):
            if cluster_keywords:
                df["keywords_in_cluster"] = df[category_column_name].map(
                    lambda c: "<br>".join(cluster_keywords.get(c, ["(no keywords found)"]))
                )
            else:
                df["keywords_in_cluster"] = "(no keywords available)"
            return df[df["count"] > 0].copy()

        df1 = enrich(df1)
        df2 = enrich(df2)

        colors = PlotGenerator.get_color_palette(df1, category_column_name)

        fig1 = px.bar_polar(
            df1, r="count", theta="year", color=category_column_name,
            template="seaborn", color_discrete_sequence=colors,
            custom_data=["keywords_in_cluster"]
        )
        fig2 = px.bar_polar(
            df2, r="count", theta="year", color=category_column_name,
            template="seaborn", color_discrete_sequence=colors,
            custom_data=["keywords_in_cluster"]
        )

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "polar"}, {"type": "polar"}]],
            subplot_titles=[title1, title2]
        )

        for trace in fig1.data:
            fig.add_trace(trace, row=1, col=1)
        for trace in fig2.data:
            trace.showlegend = False
            fig.add_trace(trace, row=1, col=2)

        fig.update_traces(hovertemplate="<b>Year:</b> %{theta}<br><b>Articles:</b> %{r}<br><b>Keywords:</b><br>%{customdata[0]}")
        fig.update_layout(
            title_text="Polar Comparison of Categories",
            title_x=0.5,
            height=height,
            width=width,
            margin=dict(t=100)
        )

        if save_image:
            out = Path(plot_folder)
            out.mkdir(parents=True, exist_ok=True)
            fig.write_image(out / filename)

        return fig

    @staticmethod
    def generate_polar_comparison_graph_split_years(
        raw_df: pd.DataFrame,
        db_handler,
        cluster_keywords: Dict[str, List[str]] = None,
        width: int = 1100,
        height: int = 650
    ) -> go.Figure:
        """
        Divides years with data in two periods
        """

        # Obtain data per category and year
        df_all_raw = db_handler.query_count_unique_papers_per_group_per_year()
        df_all = df_all_raw.rename(columns=lambda x: x.split("_")[-1] if x.startswith("paper_count_") else x)
        df_all = df_all.rename(columns={"name": "Category"})

        category_column = "Category"

        # It filters leaving only columns with some article
        year_cols = sorted([c for c in df_all.columns if c.isdigit()])
        year_cols_with_data = [y for y in year_cols if df_all[y].sum() > 0]

        if len(year_cols_with_data) < 2:
            raise ValueError("There is not enough data to create comparison polar plots.")

        # Divides years in two halves
        midpoint = len(year_cols_with_data) // 2
        years_1 = year_cols_with_data[:midpoint]
        years_2 = year_cols_with_data[midpoint:]

        meta_cols = [col for col in df_all.columns if col not in year_cols]

        def melt_years(years: list[str]) -> pd.DataFrame:
            df_sub = df_all[meta_cols + years]
            df_melted = pd.melt(df_sub, id_vars=[category_column], var_name="year", value_name="count")
            df_melted["year"] = df_melted["year"].astype(str)
            return df_melted[df_melted["count"] > 0].copy()

        df1 = melt_years(years_1)
        df2 = melt_years(years_2)
        unique_years = sorted(set(df1["year"]) | set(df2["year"]))
        print(f"[DEBUG] Años únicos disponibles: {unique_years}")
        print(df1, "df1")
        print(df2, "df2")

        return PlotGenerator.plot_polars(
            df1=df1,
            df2=df2,
            plot_folder="/tmp",
            category_column_name=category_column,
            cluster_keywords=cluster_keywords,
            title1=f"Años: {', '.join(years_1)}",
            title2=f"Años: {', '.join(years_2)}",
            save_image=False,
            width=width,
            height=height
        )


    @staticmethod
    def plot_lines(df: pd.DataFrame, category_column_name: str, plot_folder: str, filename: str = "lines_plot.pdf", colors=None, width: int = 1100, height: int = 650):
        print("lines")
        if colors is None:
            colors = PlotGenerator.get_color_palette(df, category_column_name)
        fig = px.line(
            df, x="year", y="count", color=category_column_name, markers=True,
            color_discrete_sequence=colors,
            labels={"year": "Year", "count": "Number of papers"},
            width=width, height=height
        )
        fig.update_layout(title_text="Year evolution per group category")
        out = Path(plot_folder)
        out.mkdir(parents=True, exist_ok=True)
        fig.write_image(out / filename)
        return fig

    @staticmethod
    def plot_log_lines(df: pd.DataFrame, category_column_name: str, plot_folder: str, filename: str = "log_lines_plot.pdf", colors=None, width: int = 1100, height: int = 650):
        if colors is None:
            colors = PlotGenerator.get_color_palette(df, category_column_name)
        fig = px.line(
            df, x="year", y="count", color=category_column_name, markers=True,
            color_discrete_sequence=colors, log_y=True,
            labels={"year": "Year", "count": "Number of papers"},
            width=width, height=height
        )
        fig.update_layout(title_text="Logaritmic evolution per group category")
        out = Path(plot_folder)
        out.mkdir(parents=True, exist_ok=True)
        fig.write_image(out / filename)
        return fig

    @staticmethod
    def get_country_publication_df(controller, filename: str = None) -> pd.DataFrame:
        df = controller._db_handler.query_publications_by_country_and_year()
        df.rename(columns={"country_name": "country", "publication_year": "year"}, inplace=True)
        df["year"] = df["year"].astype(str)
        df = df.sort_values(by=["year", "country"])
        total_per_year = df.groupby("year")["count"].sum().rename("total")
        df = df.join(total_per_year, on="year")
        df["norm_count"] = df["count"] / df["total"]
        if filename:
            df.to_latex(filename)
        return df


    @staticmethod
    def generate_interactive_country_map(
        df: pd.DataFrame,
        save_file_path_pdf: str,
        width: int = 1100,
        height: int = 650
    ) -> px.choropleth:
        if "country" not in df.columns or "publication_year" not in df.columns:
            raise KeyError("Columns 'Country' and 'Publication_Year' are required to display this graph.")

        # Limpieza y extracción de nombre de país
        df["country_clean"] = df["country"].str.extract(r'([A-Za-z\s]+)$').fillna(df["country"])
        df["country_clean"] = df["country_clean"].str.strip()

        # Crear la grilla completa: todos los países × todos los años
        all_years = pd.DataFrame({
            "year": range(df["publication_year"].min(), df["publication_year"].max() + 1)
        })
        all_countries = pd.DataFrame({
            "country_clean": df["country_clean"].dropna().unique()
        })
        full_grid = all_countries.merge(all_years, how="cross")

        # Agrupar datos reales
        grouped = (
            df[df["country"].notna() & (df["country"].str.strip() != "")]
            .groupby(["country_clean", "publication_year"])
            .size()
            .reset_index(name="count")
            .rename(columns={"publication_year": "year"})
        )

        merged = full_grid.merge(grouped, on=["country_clean", "year"], how="left")
        merged["count"] = merged["count"].fillna(0)

        # Normalizar por año
        merged["norm_count"] = merged.groupby("year")["count"].transform(
            lambda x: x / x.sum() if x.sum() > 0 else 0
        )

        merged["year"] = merged["year"].astype(int)
        merged = merged.sort_values(by=["year", "country_clean"])

        color_scale = [
            [0.00, "#ffffff"],   # Blanco puro para 0
            [0.20, "#e0f7e9"],   # Verde muy pálido
            [0.40, "#b3e2cc"],   # Verde claro
            [0.60, "#66c2a4"],   # Verde medio
            [0.80, "#2ca25f"],   # Verde más intenso
            [1.00, "#006d2c"]    # Verde oscuro
        ]

        # Crear gráfico interactivo
        fig = px.choropleth(
            merged,
            locations="country_clean",
            locationmode="country names",
            color="norm_count",
            animation_frame="year",
            hover_name="country_clean",
            hover_data={
                "count": True,
                "norm_count": False,
                "year": False,
                "country_clean": False
            },
            color_continuous_scale=color_scale,
            range_color=(0, merged["norm_count"].max()),
            labels={
                "count": "Publications",
                "norm_count": "Fraction",
                "year": "Year"
            },
            title="Proportion of Publications by Country Over Time"
        )

        fig.update_geos(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular'
        )
        fig.update_layout(
            width=width,
            height=height,
            margin=dict(l=50, r=50, t=80, b=50),
            coloraxis_colorbar=dict(title="Fraction")
        )

        Path(save_file_path_pdf).parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(save_file_path_pdf)

        return fig



    @staticmethod
    def generate_cumulative_country_map(df: pd.DataFrame, save_file_path_pdf: str  = None, width: int = 1100, height: int = 650 ) -> px.choropleth:
        if "country" not in df.columns or "publication_year" not in df.columns:
            raise KeyError("Columns 'Country' and 'Publication_Year' are required to display this graph.")

        df["country_clean"] = df["country"].str.extract(r'([A-Za-z\s]+)$').fillna(df["country"])
        df["country_clean"] = df["country_clean"].str.strip()

        min_year, max_year = df["publication_year"].min(), df["publication_year"].max()
        all_years = pd.DataFrame({"year": range(min_year, max_year + 1)})
        all_countries = pd.DataFrame({"country_clean": df["country_clean"].dropna().unique()})
        full_grid = all_countries.merge(all_years, how="cross")

        grouped = (
            df.groupby(["country_clean", "publication_year"])
            .size()
            .reset_index(name="count")
            .rename(columns={"publication_year": "year"})
        )

        merged = full_grid.merge(grouped, on=["country_clean", "year"], how="left")
        merged["count"] = merged["count"].fillna(0).astype(int)
        merged = merged.sort_values(by=["country_clean", "year"])

        # 5. Cumulative data calculation
        merged["cumulative_count"] = merged.groupby("country_clean")["count"].cumsum()

        color_scale = [
            [0.00, "#ffffff"],   # Blanco puro para 0
            [0.20, "#e0f7e9"],   # Verde muy pálido
            [0.40, "#b3e2cc"],   # Verde claro
            [0.60, "#66c2a4"],   # Verde medio
            [0.80, "#2ca25f"],   # Verde más intenso
            [1.00, "#006d2c"]    # Verde oscuro
        ]
        fig = px.choropleth(
            merged,
            locations="country_clean",
            locationmode="country names",
            color="cumulative_count",
            animation_frame="year",
            hover_name="country_clean",
            hover_data={
                "cumulative_count": True,
                "year": False,
                "country_clean": False
            },
            color_continuous_scale=color_scale,
            range_color=(0, merged["cumulative_count"].max()),
            labels={
                "cumulative_count": "Cumulative Publications",
                "year": "Year"
            },
            title="Cumulative Publications by Country Over Time"
        )

        fig.update_geos(showframe=False, showcoastlines=True, projection_type='equirectangular')
        fig.update_layout(
            width=width,
            height=height,
            margin=dict(l=50, r=50, t=80, b=50),
            coloraxis_colorbar=dict(title="Publications")
        )

        if save_file_path_pdf:
            Path(save_file_path_pdf).parent.mkdir(parents=True, exist_ok=True)
            fig.write_image(save_file_path_pdf)

        return fig


    def generate_interactive_author_cluster_graph(
        self,
        author_clusters: dict,
        author_citations_map: dict,
        active_authors: set,
        save_path_html: str = None,
        threshold: float = 0.0,
        layout: str = "spring",
        width: int = 1100,
        height: int = 650
    ) -> go.Figure:

        import heapq

        G = nx.Graph()
        cluster_labels = {}
        author_papers = {}
        author_citations = {}

        # Build graph
        for cluster_name, data in author_clusters.items():
            filtered_authors = [a for a in data["authors"] if a in active_authors]
            for author in filtered_authors:
                G.add_node(author)
                cluster_labels[author] = cluster_name
                author_papers[author] = author_papers.get(author, 0) + 1
                author_citations[author] = author_citations_map.get(author, 0)
            for i in range(len(filtered_authors)):
                for j in range(i + 1, len(filtered_authors)):
                    G.add_edge(filtered_authors[i], filtered_authors[j])


        # Calculate max after calculated dict
        max_articles = max(author_papers.values(), default=1)
        max_citations = max(author_citations.values(), default=0)

        # If no citations, return empty graph
        if max_citations == 0:
            return go.Figure(layout=go.Layout(
                title="Authors Semantic Clustering",
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                annotations=[dict(text="No citation data available", showarrow=False)]
            ))

        normalized_citations = {
            author: author_citations.get(author, 0) / max_citations
            for author in G.nodes()
        }

        # Filter nodes by threshold
        G = G.copy()
        for node in list(G.nodes()):
            if normalized_citations.get(node, 0) < threshold:
                G.remove_node(node)

        num_nodes = len(G.nodes())
        if num_nodes > 2000:
            label_top_n = 0
            iterations = 20
            k_value = 0.25
        elif num_nodes > 300:
            label_top_n = 30
            iterations = 40
            k_value = 0.45
        else:
            label_top_n = 100
            iterations = 100
            k_value = 0.7

        if layout == "spring":
            pos = nx.spring_layout(G, k=k_value, iterations=iterations, seed=42)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        elif layout == "shell":
            pos = nx.shell_layout(G)
        elif layout == "spectral":
            pos = nx.spectral_layout(G)
        else:
            print(f"[WARNING] Unknown layout '{layout}', falling back to spring_layout")
            pos = nx.spring_layout(G, k=k_value, iterations=iterations, seed=42)

        x_nodes = [pos[k][0] for k in G.nodes()]
        y_nodes = [pos[k][1] for k in G.nodes()]

        node_sizes = [
            6 + 24 * (author_papers.get(node, 0) / max_articles)
            for node in G.nodes()
        ]
        node_colors = [
            author_citations.get(node, 0) / max_citations
            for node in G.nodes()
        ]

        if label_top_n > 0:
            top_nodes = set(heapq.nlargest(label_top_n, author_papers, key=author_papers.get))
        else:
            top_nodes = set()

        node_labels = [
            node if node in top_nodes else ""
            for node in G.nodes()
        ]

        hover_texts = [
            f"<b>{node}</b><br>Cluster: {cluster_labels[node]}<br>Articles: {author_papers.get(node, 0)}<br>Citations: {author_citations.get(node, 0)}"
            for node in G.nodes()
        ]

        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(width=0.5, color="rgba(180,180,180,0.3)"),
            hoverinfo="none"
        )

        node_trace = go.Scatter(
            x=x_nodes, y=y_nodes,
            mode="markers+text" if label_top_n > 0 else "markers",
            text=node_labels,
            textposition="top center",
            textfont=dict(size=10),
            hoverinfo="text",
            hovertext=hover_texts,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale="Viridis",
                cmin=0,
                cmax=1,
                colorbar=dict(title="Normalized Citations"),
                opacity=0.9,
                line=dict(width=0.6, color="#ddd")
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=f"Authors Semantic Clustering ({layout} layout)",
            titlefont=dict(size=20),
            width=width,
            height=height,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=20, r=20, t=80),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#f9f9f9",
            paper_bgcolor="#f9f9f9",
        )

        if save_path_html:
            fig.write_html(save_path_html)

        return fig


    def generate_top_keywords_bubble_chart(self, author_clusters: dict, author_kw_map: dict, top_n: int = None, width: int = 1100, height: int = 650) -> go.Figure:
        bubble_data = []
        for cluster, data in author_clusters.items():
            counter = Counter()
            for author in data["authors"]:
                keywords = author_kw_map.get(author, [])
                if not keywords:
                    continue
                counter.update(keywords)

            keywords = counter.most_common(top_n) if top_n else counter.items()
            for kw, freq in keywords:
                bubble_data.append({
                    "Cluster": cluster,
                    "Keyword": kw,
                    "Frequency": freq
                })

        df_bubble = pd.DataFrame(bubble_data)
        if df_bubble.empty or not {"Cluster", "Keyword", "Frequency"}.issubset(df_bubble.columns):
            return Figure(layout=go.Layout(
                title="Top Keywords per Cluster of Authors",
                annotations=[dict(text="No data available to display", showarrow=False)],
                xaxis=dict(visible=False), yaxis=dict(visible=False)
            ))

        fig = px.scatter(
            df_bubble,
            x="Cluster",
            y="Keyword",
            size="Frequency",
            color="Cluster",
            size_max=60,
            title="Top Keywords per Cluster of Authors",
            width=width,
            height=height
        )

        fig.update_traces(
            marker=dict(opacity=0.7, line=dict(width=1, color='DarkSlateGrey'))
        )
        fig.update_layout(
            xaxis_title="Cluster of Authors",
            yaxis_title="Keyword",
            hovermode="closest"
        )
        return fig

    def generate_interactive_author_keyword_heatmap(
        self,
        author_clusters: dict,
        author_kw_map: dict,
        save_path_html: str = None,
        min_freq: int = 0, width: int = 1100, height: int = 650
    ) -> go.Figure:

        heatmap_data = {}
        all_keywords = set()

        for cluster_name, data in author_clusters.items():
            keyword_counter = Counter()
            for author in data["authors"]:
                keyword_counter.update(author_kw_map.get(author.strip(), []))
            if min_freq > 0:
                keyword_counter = {kw: cnt for kw, cnt in keyword_counter.items() if cnt >= min_freq}
            heatmap_data[cluster_name] = keyword_counter
            all_keywords.update(keyword_counter.keys())

        all_keywords = sorted(list(all_keywords))
        if not all_keywords:
            return go.Figure(layout=go.Layout(
                title="Heatmap of Keywords per Cluster of Authors",
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                annotations=[dict(text="No data to display", showarrow=False)]
            ))

        df_heatmap = pd.DataFrame(index=all_keywords, columns=author_clusters.keys()).fillna(0)

        for cluster_name in author_clusters:
            for kw in all_keywords:
                df_heatmap.loc[kw, cluster_name] = heatmap_data[cluster_name].get(kw, 0)

        fig = go.Figure(data=go.Heatmap(
            z=df_heatmap.values,
            x=df_heatmap.columns,
            y=df_heatmap.index,
            colorscale='YlOrBr',
            colorbar=dict(title="Frecuencia"),
            hovertemplate="Cluster: %{x}<br>Keyword: %{y}<br>Frequency: %{z}<extra></extra>",
            zmin=0,
            zmax=df_heatmap.values.max(),
        ))

        fig.update_layout(
            title="Heatmap of Keywords per Cluster of Authors",
            xaxis_title="Cluster",
            yaxis_title="Keyword",
            autosize=False,
            width=width,
            height=height,
            margin=dict(l=100, r=100, t=80, b=80),
            xaxis_tickangle=45,
            font=dict(size=10),
        )

        if save_path_html:
            from pathlib import Path
            Path(save_path_html).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(save_path_html)

        return fig


    def export_plotly_animation_as_gif(self, fig, output_path, fps=3, width=1100, height=650):

        temp_folder = "temp_frames"
        os.makedirs(temp_folder, exist_ok=True)

        frame_paths = []
        for i, frame in enumerate(fig.frames):
            fig.update(data=frame.data)
            frame_path = os.path.join(temp_folder, f"frame_{i:04d}.png")
            pio.write_image(fig, frame_path, width=width, height=height, engine="kaleido")
            frame_paths.append(frame_path)

        images = [imageio.imread(fp) for fp in frame_paths]
        imageio.mimsave(output_path, images, fps=fps)

        for fp in frame_paths:
            os.remove(fp)
        os.rmdir(temp_folder)

    @staticmethod
    def generate_citation_heatmap_by_country(
        df: pd.DataFrame,
        save_path_html: str = None,
        width: int = 1100,
        height: int = 650
    ) -> go.Figure:

        if "country" not in df.columns or "citations" not in df.columns or "publication_year" not in df.columns:
            raise KeyError("Se requieren las columnas 'country', 'citations' y 'publication_year' en el DataFrame.")

        # Limpiar y extraer país
        df["country_clean"] = df["country"].str.extract(r'([A-Za-z\s]+)$').fillna(df["country"])
        df["country_clean"] = df["country_clean"].str.strip()

        # Crear grilla completa: todos los países × todos los años
        all_years = pd.DataFrame({
            "year": range(df["publication_year"].min(), df["publication_year"].max() + 1)
        })
        all_countries = pd.DataFrame({
            "country_clean": df["country_clean"].dropna().unique()
        })
        full_grid = all_countries.merge(all_years, how="cross")

        # Agrupar datos reales
        grouped = (
            df[df["country"].notna() & (df["country"].str.strip() != "")]
            .groupby(["country_clean", "publication_year"])["citations"]
            .sum()
            .reset_index()
            .rename(columns={"publication_year": "year"})
        )

        # Mezclar con la grilla para forzar todos los años y países
        merged = full_grid.merge(grouped, on=["country_clean", "year"], how="left")
        merged["citations"] = merged["citations"].fillna(0)

        # Normalizar por año
        merged["norm_citations"] = merged.groupby("year")["citations"].transform(
            lambda x: x / x.sum() if x.sum() > 0 else 0
        )

        merged["year"] = merged["year"].astype(int)
        merged = merged.sort_values(by=["year", "country_clean"])

        # Escala de color profesional
        color_scale = [
            [0.00, "#ffffff"],   # Blanco puro para 0
            [0.20, "#e0f7e9"],   # Verde muy pálido
            [0.40, "#b3e2cc"],   # Verde claro
            [0.60, "#66c2a4"],   # Verde medio
            [0.80, "#2ca25f"],   # Verde más intenso
            [1.00, "#006d2c"]    # Verde oscuro
        ]
        # Crear gráfico
        fig = px.choropleth(
            merged,
            locations="country_clean",
            locationmode="country names",
            color="norm_citations",
            hover_name="country_clean",
            hover_data={
                "citations": True,
                "year": False,
                "norm_citations": False,
                "country_clean": False
            },
            animation_frame="year",
            color_continuous_scale=color_scale,
            range_color=(0, merged["norm_citations"].max()),
            labels={
                "norm_citations": "Fraction",
                "citations": "Citations",
                "year": "Year"
            },
            title="Proportion of Citations by Country Over Time"
        )

        fig.update_geos(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular'
        )
        fig.update_layout(
            width=width,
            height=height,
            margin=dict(l=50, r=50, t=80, b=50),
            coloraxis_colorbar=dict(title="Frac."),
            geo=dict(showcountries=True),
        )

        if save_path_html:
            Path(save_path_html).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(save_path_html)

        return fig


    @staticmethod
    def generate_cumulative_citation_map_by_country(
        df: pd.DataFrame,
        save_path_html: str = None, width: int = 1100, height: int = 650
    ) -> go.Figure:

        if "country" not in df.columns or "citations" not in df.columns or "publication_year" not in df.columns:
            raise KeyError("Se requieren las columnas 'country', 'citations' y 'publication_year' en el DataFrame.")

        df["country_clean"] = df["country"].str.extract(r'([A-Za-z\s]+)$').fillna(df["country"])
        df["country_clean"] = df["country_clean"].str.strip()

        all_years = pd.DataFrame({"year": range(df["publication_year"].min(), df["publication_year"].max() + 1)})
        all_countries = pd.DataFrame({"country_clean": df["country_clean"].dropna().unique()})
        full_grid = all_countries.merge(all_years, how="cross")  #

        grouped = (
            df[df["country"].notna() & (df["country"].str.strip() != "")]
            .groupby(["country_clean", "publication_year"])["citations"]
            .sum()
            .reset_index()
            .rename(columns={"publication_year": "year"})
        )

        merged = full_grid.merge(grouped, on=["country_clean", "year"], how="left")
        merged["citations"] = merged["citations"].fillna(0)
        merged["year"] = merged["year"].astype(int)

        merged = merged.sort_values(by=["country_clean", "year"])
        merged["cumulative_citations"] = merged.groupby("country_clean")["citations"].cumsum()

        color_scale = [
            [0.00, "#ffffff"],   # Blanco puro para 0
            [0.20, "#e0f7e9"],   # Verde muy pálido
            [0.40, "#b3e2cc"],   # Verde claro
            [0.60, "#66c2a4"],   # Verde medio
            [0.80, "#2ca25f"],   # Verde más intenso
            [1.00, "#006d2c"]    # Verde oscuro
        ]
        fig = px.choropleth(
            merged,
            locations="country_clean",
            locationmode="country names",
            color="cumulative_citations",
            animation_frame="year",
            hover_name="country_clean",
            hover_data={
                "cumulative_citations": True,
                "year": False,
                "country_clean": False
            },
            color_continuous_scale=color_scale,
            range_color=(0, merged["cumulative_citations"].max()),
            labels={
                "cumulative_citations": "Cumulative Citations",
                "year": "Year"
            },
            title="Cumulative Citations by Country Over Time"
        )

        fig.update_geos(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular'
        )
        fig.update_layout(
            width=width,
            height=height,
            margin=dict(l=50, r=50, t=80, b=50),
            coloraxis_colorbar=dict(title="Citations"),
            geo=dict(showcountries=True),
        )

        if save_path_html:
            Path(save_path_html).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(save_path_html)

        return fig

    @staticmethod
    def generate_interactive_coauthorship_graph(
        raw_df: pd.DataFrame,
        save_file_path_pdf: str,
        width: int = 1100, height: int = 650,
        top_label_threshold: float = 0.9,
        threshold: float = 0.0,
        layout: str = "spring"  # Nuevo parámetro
    ) -> go.Figure:

        def split_authors(s: str) -> list[str]:
            if pd.isna(s):
                return []
            return [a.strip() for a in s.split(";") if a.strip()]

        df = raw_df.copy()
        df["_authors_"] = df["author"].apply(split_authors)

        # Construir grafo
        G = nx.Graph()
        for authors in df["_authors_"]:
            for a in authors:
                prev = G.nodes[a]["papers"] if G.has_node(a) else 0
                G.add_node(a, papers=prev + 1)
            for i in range(len(authors)):
                for j in range(i + 1, len(authors)):
                    u, v = authors[i], authors[j]
                    w = G[u][v]["weight"] + 1 if G.has_edge(u, v) else 1
                    G.add_edge(u, v, weight=w)

        # Elegir layout
        if layout == "spring":
            pos = nx.spring_layout(G, seed=42, k=1.2, iterations=100)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        elif layout == "shell":
            pos = nx.shell_layout(G)
        elif layout == "spectral":
            pos = nx.spectral_layout(G)
        else:
            print(f"[WARNING] Unknown layout '{layout}', falling back to spring_layout")
            pos = nx.spring_layout(G, seed=42, k=1.2, iterations=100)

        # Filtrar nodos por threshold
        node_x, node_y, node_text, node_size, node_color, node_label = [], [], [], [], [], []
        visible_nodes_set = set()
        degrees = dict(G.degree())
        maxdeg = max(degrees.values()) or 1

        for n in G.nodes():
            x, y = pos[n]
            papers = G.nodes[n]["papers"]
            deg = degrees[n]
            norm_deg = deg / maxdeg

            if norm_deg < threshold:
                continue

            visible_nodes_set.add(n)
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{n}<br>#papers: {papers}<br>Connections: {deg}")
            node_size.append(min(20 + papers * 2, 60))
            node_color.append(norm_deg)
            node_label.append(n if norm_deg >= top_label_threshold else "")

        # Construir aristas
        edge_x, edge_y = [], []
        for u, v in G.edges():
            if u not in visible_nodes_set or v not in visible_nodes_set:
                continue
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        # Crear figura Plotly
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(width=0.5, color="#999"),
            hoverinfo="none",
        )

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=node_label,
            textposition="top center",
            textfont=dict(size=9),
            hoverinfo="text",
            hovertext=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale="Blues",
                showscale=True,
                colorbar=dict(title="Normalized Degree"),
                line_width=1,
            ),
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=f"Co-authorship Network ({layout} layout)",
            title_font=dict(size=20),
            width=width,
            height=height,
            showlegend=False,
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#f9f9f9",
            paper_bgcolor="#f9f9f9",
        )

        Path(save_file_path_pdf).parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(save_file_path_pdf)

        return fig

