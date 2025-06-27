import sqlite3 ##to work with database
import os
import pandas as pd


class DBHandler:


    def __init__(self, db_path: str = None) -> None:
        """
        If db_path is provided, we use that path.
        If not, for compatibility, we continue using Review.db in the cwd.
        """
        self._db_name: str = db_path or "Review.db"

    def limit_query(self, query: str, limit: int = None) -> str:
        if limit is not None and limit > 0:
            query += f" LIMIT {limit}"
        return query

    def build_query(self, query: str, limit: int = None) -> str:
        query = self.limit_query(query, limit) + ";"
        return query

    def get_unique_years(self, cursor: sqlite3.Cursor) -> list[int]:
        cursor.execute(
            "SELECT DISTINCT publication_year FROM Paper ORDER BY publication_year ASC"
        )
        years = [int(row[0]) for row in cursor.fetchall()]
        return years

    def is_database_created(self) -> bool:
        print(os.path.isfile(self._db_name))
        return os.path.isfile(self._db_name)
    

    def delete_database(self) -> bool:
        if self.is_database_created():
            os.remove(self._db_name)
            return True
        return False

    def create_database(self) -> None:
        conn = sqlite3.connect(self._db_name)
        cursor = conn.cursor()

        # Create Paper table
        cursor.execute(
            """CREATE TABLE Paper (
                            paper_id INTEGER PRIMARY KEY,
                            title TEXT NOT NULL,
                            publication_year INTEGER
                        )"""
        )

        # Create KeywordGroup table
        cursor.execute(
            """CREATE TABLE KeywordGroup (
                        group_id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        parent_group_id INTEGER,
                        FOREIGN KEY (parent_group_id) REFERENCES KeywordGroup(group_id)
                    )"""
        )

        # Create Keyword table
        cursor.execute(
            """CREATE TABLE Keyword (
                            keyword_id INTEGER PRIMARY KEY,
                            name TEXT NOT NULL,
                            group_id INTEGER,
                            FOREIGN KEY (group_id) REFERENCES KeywordGroup(group_id)
                        )"""
        )

        # Create Paper_Keyword table
        cursor.execute(
            """CREATE TABLE Paper_Keyword (
                            paper_id INTEGER,
                            keyword_id INTEGER,
                            PRIMARY KEY (paper_id, keyword_id),
                            FOREIGN KEY (paper_id) REFERENCES Paper(paper_id),
                            FOREIGN KEY (keyword_id) REFERENCES Keyword(keyword_id)
                        )"""
        )

        # Create Author_Cluster table
        cursor.execute(
            """CREATE TABLE Paper_Author (
                paper_id INTEGER,
                author TEXT,
                cluster_id INTEGER,
                PRIMARY KEY (paper_id, author),
                FOREIGN KEY (paper_id) REFERENCES Paper(paper_id)
            )"""
        )


        # Commit changes and close connection
        conn.commit()
        conn.close()

    def populate_paper_table(self, df: pd.DataFrame):
        column_names = ["publication_year", "title"]
        print(df)
        if df is None:
            return 

        # Connect to the SQLite database
        conn = sqlite3.connect(self._db_name)
        print("table paper")
        print(df[column_names].head())
        # Insert DataFrame records into the Paper table
        df[column_names].to_sql("Paper", conn, if_exists="append", index=False)
        # Commit changes and close connection
        conn.commit()
        conn.close()

    def _populate_keyword_group_table(
        self, cursor: sqlite3.Cursor, keyword_semantic_goups: dict[str, list[str]]
    ):
        # Create KeywordGroup table and populate it
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS KeywordGroup (
                        group_id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        parent_group_id INTEGER,
                        FOREIGN KEY (parent_group_id) REFERENCES KeywordGroup(group_id)
                    )"""
        )

        for group_id, group_name in enumerate(keyword_semantic_goups.keys(), start=1):
            print("group_id, group_name", group_id, group_name)
            cursor.execute(
                """INSERT INTO KeywordGroup (group_id, name) VALUES (?, ?)""",
                (group_id, group_name),
            )

    def _populate_keyword_table(
        self, cursor: sqlite3.Cursor, keyword_semantic_goups: dict[str, list[str]]
    ):
        # Create Keyword table and populate it
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS Keyword (
                            keyword_id INTEGER PRIMARY KEY,
                            name TEXT NOT NULL,
                            group_id INTEGER,
                            FOREIGN KEY (group_id) REFERENCES KeywordGroup(group_id)
                        )"""
        )

        for group_name, keywords in keyword_semantic_goups.items():
            group_id = cursor.execute(
                """SELECT group_id FROM KeywordGroup WHERE name = ?""", (group_name,)
            ).fetchone()[0]
            for keyword in keywords:
                cursor.execute(
                    """INSERT INTO Keyword (name, group_id) VALUES (?, ?)""",
                    (keyword, group_id),
                )

    def _clear_keyword_tables(self, cursor: sqlite3.Cursor):
        # Delete all content from KeywordGroup table
        cursor.execute("DELETE FROM KeywordGroup")
        # Update references in Keyword table
        cursor.execute("UPDATE Keyword SET group_id = NULL")

    def _populate_keyword_group_table_updating_keyword_table(
        self, cursor: sqlite3.Cursor, keyword_semantic_goups: dict[str, list[str]]
    ): ## If a group already exists, its group_id is retrieved; otherwise, it is inserted.
        # Populate KeywordGroup and update Keyword tables
        for group_name, keywords in keyword_semantic_goups.items():
            # Insert or retrieve group_id for the group name
            cursor.execute(
                "SELECT group_id FROM KeywordGroup WHERE name = ?", (group_name,)
            )
            row = cursor.fetchone()
            if row:
                group_id = row[0]
            else:
                cursor.execute(
                    "INSERT INTO KeywordGroup (name) VALUES (?)", (group_name,)
                )
                group_id = cursor.lastrowid

            # Update the group_id for keywords
            for keyword in keywords: ## Updates the records in the Keyword table to assign the correct group_id to each keyword.
                cursor.execute(
                    "UPDATE Keyword SET group_id = ? WHERE name = ?",
                    (group_id, keyword),
                )

    def regenerate_keyword_tables(self, keyword_semantic_goups: dict[str, list[str]]):
        conn = sqlite3.connect(self._db_name)
        cursor = conn.cursor()

        self._clear_keyword_tables(cursor)
        self._populate_keyword_group_table_updating_keyword_table(
            cursor, keyword_semantic_goups
        )

        # Commit changes and close connection
        conn.commit()
        conn.close()

    def populate_keyword_tables(self, keyword_semantic_goups: dict[str, list[str]]):
        conn = sqlite3.connect(self._db_name)
        cursor = conn.cursor()

        # Populate KeywordGroup and Keyword tables
        self._populate_keyword_group_table(cursor, keyword_semantic_goups)
        self._populate_keyword_table(cursor, keyword_semantic_goups)

        # Commit changes and close connection
        conn.commit()
        conn.close()

    def populate_paper_keyword_table(
        self,
        df: pd.DataFrame,
    ) -> None:
        # Connect to the SQLite database
        conn = sqlite3.connect(self._db_name)
        cursor = conn.cursor()
        print(df["keywords"], "djcenra ckajk")
        print("Ejemplo de keywords en DataFrame:", df["keywords"].iloc[0])
        # Populate Paper_Keyword table
        for paper_id, keywords in enumerate(df["keywords"], start=1):
            for keyword in keywords:
                keyword_norm = keyword.strip().lower()
                keyword_id = cursor.execute(
                    """SELECT keyword_id FROM Keyword WHERE name = ?""",
                    (keyword_norm,)
                ).fetchone()
                if keyword_id:
                    keyword_id = keyword_id[0]
                    cursor.execute(
                        """INSERT INTO Paper_Keyword (paper_id, keyword_id) VALUES (?, ?)""",
                        (paper_id, keyword_id),
                    )
        conn.commit()
        conn.close()

    def query_count_unique_papers_per_group(self, limit: int = None) -> pd.DataFrame:
        result: pd.DataFrame = None

        # Connect to the SQLite database
        conn = sqlite3.connect(self._db_name)

        # SQL query to retrieve the top 30 groups with the most unique papers
        query = """
            SELECT KeywordGroup.name, COUNT(DISTINCT Paper.paper_id) AS paper_count
            FROM KeywordGroup
            JOIN Keyword ON Keyword.group_id = KeywordGroup.group_id
            JOIN Paper_Keyword ON Keyword.keyword_id = Paper_Keyword.keyword_id
            JOIN Paper ON Paper_Keyword.paper_id = Paper.paper_id
            GROUP BY KeywordGroup.name
            ORDER BY paper_count DESC
        """

        query = self.build_query(query, limit)

        # Fetch all results
        result = pd.read_sql_query(query, conn)

        # Close connection
        conn.close()

        return result



    def query_count_unique_papers_per_group_per_year(
        self, limit: int = None
    ) -> pd.DataFrame:
        result: pd.DataFrame = None
        """
        First, it obtains the total number of articles per group.
        Then, for each year (obtained using get_unique_years), it builds a query that retrieves the count of articles for that year.
        Finally, it merges the results to form a DataFrame that contains the year-by-year evolution of each group.
        """
        # Connect to the SQLite database
        conn = sqlite3.connect(self._db_name)
        cursor = conn.cursor()

        unique_years: list[int] = self.get_unique_years(cursor)

        # Construct the SQL query dynamically
        query = """
            SELECT 
                KeywordGroup.name,
                COUNT(DISTINCT Paper.paper_id) AS total_paper_count
            FROM KeywordGroup
            JOIN Keyword ON Keyword.group_id = KeywordGroup.group_id
            JOIN Paper_Keyword ON Keyword.keyword_id = Paper_Keyword.keyword_id
            JOIN Paper ON Paper_Keyword.paper_id = Paper.paper_id
            GROUP BY KeywordGroup.name
            ORDER BY total_paper_count DESC
        """

        query = self.build_query(query, limit)

        result = pd.read_sql_query(query, conn)

        for year in unique_years:
            query = f"""
                SELECT 
                    KeywordGroup.name,
                    COUNT(DISTINCT Paper.paper_id) AS paper_count_{year}
                FROM KeywordGroup
                JOIN Keyword ON Keyword.group_id = KeywordGroup.group_id
                JOIN Paper_Keyword ON Keyword.keyword_id = Paper_Keyword.keyword_id
                JOIN Paper ON Paper_Keyword.paper_id = Paper.paper_id
                WHERE Paper.publication_year = {year}
                GROUP BY KeywordGroup.name;
            """
            result_year = pd.read_sql_query(query, conn)
            result = result.merge(result_year, how="left", on="name")
            result = result.fillna(0)

        # Close connection
        conn.close()

        result = result.astype(
            dtype={"name": str, **{col: int for col in result.columns if col != "name"}}
        )
        return result

    def query_tendencies_of_keywords(self, limit: int = None) -> pd.DataFrame:
        result: pd.DataFrame = None

        # Connect to the SQLite database
        conn = sqlite3.connect(self._db_name)
        cursor = conn.cursor()

        unique_years: list[int] = self.get_unique_years(cursor)

        # Construct the SQL query dynamically
        query = f"""
            SELECT KeywordGroup.name,
                COUNT(DISTINCT Paper.paper_id) AS total_paper_count,
                {''.join([f"SUM(CASE WHEN Paper.publication_year = {year} THEN 1 ELSE 0 END) AS paper_count_{year}, " for year in unique_years])[:-2]}
            FROM KeywordGroup
            JOIN Keyword ON Keyword.group_id = KeywordGroup.group_id
            JOIN Paper_Keyword ON Keyword.keyword_id = Paper_Keyword.keyword_id
            JOIN Paper ON Paper_Keyword.paper_id = Paper.paper_id
            GROUP BY KeywordGroup.name
            ORDER BY total_paper_count DESC
        """

        query = self.build_query(query, limit)

        result = pd.read_sql_query(query, conn)

        # Close connection
        conn.close()

        return result
    def populate_paper_author_table(self, df: pd.DataFrame, author_clusters: dict[str, int]) -> None:
        """
        Creates and populates the Paper_Author table by splitting the authors of each paper and assigning their cluster_id.
        """
        conn = sqlite3.connect(self._db_name)
        cursor = conn.cursor()

        # Create table if it does not exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Paper_Author (
                paper_id INTEGER,
                author TEXT,
                cluster_id INTEGER,
                FOREIGN KEY (paper_id) REFERENCES Paper(paper_id)
            )
        """)

        for paper_id, row in enumerate(df.itertuples(index=False), start=1):
            authors_str = getattr(row, "author", "") or ""
            authors = [a.strip() for a in authors_str.split(";") if a.strip()]
            # Delete duplicates
            unique_authors = set(authors)
            for author in unique_authors:
                cluster_id = author_clusters.get(author)
                cursor.execute(
                    "INSERT INTO Paper_Author (paper_id, author, cluster_id) VALUES (?, ?, ?)",
                    (paper_id, author, cluster_id)
                )


        conn.commit()
        conn.close()

    def query_top_groups(
        self,
        limit: int = 10,
        year_lower_bound: int = 2024,
        year_upper_bound: int = 2024,
        excluded_keywords: list[str] = None,
    ) -> pd.DataFrame:
        """
        Generates a pandas.DataFrame with the keyword group name ("name" column) and
        the unique paper count ("unique_paper_count" column) of the `limit` groups with
        more number of unique paper counts. The search can be filtered by year and by
        `excluded_keywords` list.

        Args:
            limit (int, optional): The number of top keyword groups returning. Defaults to 10.
            year_lower_bound (int, optional): Lower bound of year filtering (number included in the search). Defaults to 2024.
            year_upper_bound (int, optional): Upper bound of year filtering (number included in the search). Defaults to 2024.
            excluded_keywords (list[str], optional): List of excluded keyword groups. Defaults to [].

        Returns:
            pd.DataFrame: The pandas.DataFrame with columns "name" for the keyword groups
            and "unique_paper_count" for the number of unique papers per group.
        """
        query = f"""
            SELECT KeywordGroup.name, COUNT(DISTINCT Paper.paper_id) AS unique_paper_count
            FROM KeywordGroup
            JOIN Keyword ON Keyword.group_id = KeywordGroup.group_id
            JOIN Paper_Keyword ON Keyword.keyword_id = Paper_Keyword.keyword_id
            JOIN Paper ON Paper_Keyword.paper_id = Paper.paper_id
            WHERE Paper.publication_year >= {year_lower_bound}
            AND Paper.publication_year <= {year_upper_bound}
            AND KeywordGroup.name NOT IN ({', '.join([f'"{group}"' for group in excluded_keywords])})
            GROUP BY KeywordGroup.name
            ORDER BY unique_paper_count DESC
        """
        if not excluded_keywords:
            excluded_keywords = []

        query = self.build_query(query, limit)

        conn = sqlite3.connect(self._db_name)
        result = pd.read_sql_query(query, conn)
        conn.close()
        return result

    def query_trends_of_groups(
        self,
        df: pd.DataFrame,
        year_lower_bound: int = 0,
        year_upper_bound: int = 3000,
        excluded_keywords: list[str] = None,
    ) -> pd.DataFrame:
        """
        Generates a pandas.DataFrame with the keyword group name ("name" column), the
        publication year of the papers of the group ("publication_year" column) and the
        unique paper count ("unique_paper_count" column) of the groups from the df DataFrame
        to see their tendencies over the years. The search can be filtered by `excluded_keywords` list.

        Args:
            df (pd.DataFrame): Keyword groups pandas.DataFrame with the group names in the column "name".
            year_lower_bound (int, optional): Lower bound of year filtering (number included in the search). Defaults to 0.
            year_upper_bound (int, optional): Upper bound of year filtering (number included in the search). Defaults to 3000.
            excluded_keywords (list[str], optional): List of excluded keyword groups. Defaults to [].

        Returns:
            pd.DataFrame: The pandas.DataFrame with columns "name" for the keyword groups,
            "publication_year" column for the year of publication and the
            "unique_paper_count" column for the number of unique papers per group per year.
        """
        if not excluded_keywords:
            excluded_keywords = []

        query = f"""
            SELECT KeywordGroup.name, Paper.publication_year, COUNT(DISTINCT Paper.paper_id) AS unique_paper_count
            FROM KeywordGroup
            JOIN Keyword ON Keyword.group_id = KeywordGroup.group_id
            JOIN Paper_Keyword ON Keyword.keyword_id = Paper_Keyword.keyword_id
            JOIN Paper ON Paper_Keyword.paper_id = Paper.paper_id
            WHERE KeywordGroup.name IN ({', '.join([f'"{group}"' for group in df['name'] if group not in excluded_keywords])})
            AND Paper.publication_year >= {year_lower_bound}
            AND Paper.publication_year <= {year_upper_bound}
            GROUP BY KeywordGroup.name, Paper.publication_year
            ORDER BY unique_paper_count, KeywordGroup.name, Paper.publication_year
        """

        query = self.build_query(query)
        conn = sqlite3.connect(self._db_name)
        result = pd.read_sql_query(query, conn)
        conn.close()
        return result

    def query_publications_by_country_and_year(self) -> pd.DataFrame:
        conn = sqlite3.connect(self._db_name)
        query = """
            SELECT country AS country_name, publication_year, COUNT(*) AS count
            FROM Paper
            WHERE country IS NOT NULL AND TRIM(country) != ''
            GROUP BY country, publication_year
            ORDER BY publication_year ASC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    def query_total_citations_by_country(self) -> pd.DataFrame:
        query = """
            SELECT country AS country, SUM(citations) AS citations
            FROM Paper
            WHERE country IS NOT NULL AND TRIM(country) != ''
            GROUP BY country
            ORDER BY citations DESC
        """
        conn = sqlite3.connect(self._db_name)
        result = pd.read_sql_query(query, conn)
        conn.close()
        return result

    def query_all_papers(self) -> pd.DataFrame:
        conn = sqlite3.connect(self._db_name)
        df = pd.read_sql_query("SELECT * FROM Paper", conn)
        conn.close()
        return df
