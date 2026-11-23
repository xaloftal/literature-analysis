"""
Reusable Bibliometric & Corpus Analysis Tool
============================================
Inspired by VOSviewer (co-occurrence tables & overlay visualizations,
co-authorship, citation/venue analytics) and Voyant Tools (Cirrus word clouds,
Corpus Summary, Term Trends over time, KWIC Contexts, Collocates, and N-grams).

Images are saved automatically to `output/<csv_stem>/` by default.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterable, Optional, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud

sns.set_theme(style="whitegrid", context="notebook")

DEFAULT_COL_MAP = {
    "title": "Document Title",
    "authors": "Authors",
    "affiliations": "Author Affiliations",
    "venue": "Publication Title",
    "year": "Publication Year",
    "abstract": "Abstract",
    "author_keywords": "Author Keywords",
    "ieee_terms": "IEEE Terms",
    "mesh_terms": "Mesh_Terms",
    "citations": "Article Citation Count",
    "references": "Reference Count",
    "doc_type": "Document Identifier",
}

DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "should", "shouldn't", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "were", "weren't", "what", "when", "where", "which", "while",
    "who", "whom", "why", "with", "won't", "would", "wouldn't", "you", "your",
    "yours", "yourself", "yourselves",
    # Academic / abstract boilerplate
    "using", "used", "use", "based", "paper", "propose", "proposed", "study",
    "results", "show", "shows", "shown", "approach", "method", "methods",
    "system", "systems", "also", "however", "well", "new", "one", "two",
    "first", "second", "within", "among", "across", "including", "provide",
    "provides", "provided", "present", "presents", "presented", "work",
    "research", "analysis", "data", "different", "various", "high", "low",
    "part", "may", "will", "et", "al", "ieee", "iso", "iec",
}


class LiteratureAnalyzer:
    """All-in-one bibliometric (VOSviewer-like) & text (Voyant-like) analyzer."""

    def __init__(
        self,
        data: Union[str, Path, pd.DataFrame],
        col_map: Optional[dict[str, str]] = None,
        extra_stopwords: Optional[Iterable[str]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        if isinstance(data, pd.DataFrame):
            self.df = data.copy()
            default_out = Path("output") / "dataset"
        else:
            data_path = Path(data)
            self.df = pd.read_csv(data_path)
            default_out = Path("output") / data_path.stem

        self.cols = {**DEFAULT_COL_MAP, **(col_map or {})}
        self.stopwords = set(DEFAULT_STOPWORDS)
        if extra_stopwords:
            self.stopwords.update(w.lower().strip() for w in extra_stopwords)

        self.output_dir = Path(output_dir) if output_dir else default_out
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self._prepare_data()

    def _col(self, key: str) -> Optional[str]:
        col_name = self.cols.get(key)
        return col_name if col_name and col_name in self.df.columns else None

    def _prepare_data(self) -> None:
        year_col = self._col("year")
        if year_col:
            self.df[year_col] = pd.to_numeric(self.df[year_col], errors="coerce")

        for num_key in ("citations", "references"):
            c = self._col(num_key)
            if c:
                self.df[c] = pd.to_numeric(self.df[c], errors="coerce")

        title_col = self._col("title")
        abs_col = self._col("abstract")
        title_s = self.df[title_col].fillna("") if title_col else pd.Series("", index=self.df.index)
        abs_s = self.df[abs_col].fillna("") if abs_col else pd.Series("", index=self.df.index)
        self.df["_corpus_text"] = (title_s + " " + abs_s).str.strip()

    def _save_fig(self, fig: plt.Figure, filename: Optional[str]) -> None:
        if self.output_dir and filename:
            fig.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")

    # ------------------------------------------------------------------
    # 1. BIBLIOMETRIC OVERVIEW & CITATION / VENUE ANALYTICS
    # ------------------------------------------------------------------
    def summary(self) -> pd.DataFrame:
        """Return high-level bibliometric indicators as a clean table."""
        year_col = self._col("year")
        cit_col = self._col("citations")
        doc_col = self._col("doc_type")
        auth_col = self._col("authors")

        rows = [("Total Documents", len(self.df))]
        if year_col:
            rows.append(("Publication Year Range", f"{int(self.df[year_col].min())} - {int(self.df[year_col].max())}"))
        if cit_col:
            cits = self.df[cit_col].fillna(0).astype(int)
            sorted_cits = np.sort(cits.values)[::-1]
            h_index = int(np.sum(sorted_cits >= np.arange(1, len(sorted_cits) + 1)))
            rows.extend([
                ("Total Citations", int(cits.sum())),
                ("Mean Citations / Doc", round(float(cits.mean()), 2)),
                ("Median Citations / Doc", float(cits.median())),
                ("Dataset h-index", h_index),
            ])
        if auth_col:
            all_authors = self._parse_semicolon_list(auth_col)
            flat_authors = [a for sub in all_authors for a in sub]
            rows.append(("Unique Authors", len(set(flat_authors))))
        if doc_col:
            rows.append(("Document Types Count", int(self.df[doc_col].nunique())))

        return pd.DataFrame(rows, columns=["Metric", "Value"])

    def document_types_table(self) -> pd.DataFrame:
        """Breakdown of documents by Document Identifier."""
        doc_col = self._col("doc_type")
        if not doc_col:
            raise ValueError("Document Identifier column not found.")
        vc = self.df[doc_col].value_counts()
        return pd.DataFrame({
            "Document Type": vc.index,
            "Count": vc.values,
            "Share (%)": (vc.values / len(self.df) * 100).round(2),
        })

    def plot_publications_and_citations_over_time(
        self, figsize: tuple[int, int] = (13, 5), save_as: Optional[str] = None
    ) -> plt.Figure:
        """Plot annual publication volume (by doc type if available) and citations."""
        year_col = self._col("year")
        if not year_col:
            raise ValueError("Publication Year column not found.")

        cit_col = self._col("citations")
        doc_col = self._col("doc_type")

        fig, axes = plt.subplots(1, 2 if cit_col else 1, figsize=figsize)
        if not cit_col:
            axes = [axes]

        ax1 = axes[0]
        if doc_col:
            top_types = self.df[doc_col].value_counts().head(5).index
            doc_grouped = self.df[doc_col].where(self.df[doc_col].isin(top_types), "Other")
            ct = pd.crosstab(self.df[year_col].dropna().astype(int), doc_grouped.loc[self.df[year_col].notna()])
            ct.plot(kind="bar", stacked=True, ax=ax1, colormap="tab10", width=0.8)
            ax1.legend(title="Document Type", fontsize=8, title_fontsize=9)
        else:
            counts = self.df[year_col].dropna().astype(int).value_counts().sort_index()
            counts.plot(kind="bar", ax=ax1, color="#4C72B0", width=0.8)

        ax1.set_title("Publications per Year", fontweight="bold")
        ax1.set_xlabel("Publication Year")
        ax1.set_ylabel("Number of Documents")
        ticks = ax1.get_xticklabels()
        if len(ticks) > 15:
            step = max(1, len(ticks) // 12)
            for idx, lbl in enumerate(ticks):
                if idx % step != 0:
                    lbl.set_visible(False)
        ax1.tick_params(axis="x", rotation=45)

        if cit_col:
            ax2 = axes[1]
            yearly = (
                self.df.dropna(subset=[year_col])
                .assign(_y=lambda d: d[year_col].astype(int))
                .groupby("_y")[cit_col]
                .agg(total="sum", mean="mean")
            )
            ax2.bar(yearly.index, yearly["total"], color="#55A868", alpha=0.7, label="Total Citations")
            ax2.set_title("Citations by Publication Year", fontweight="bold")
            ax2.set_xlabel("Publication Year")
            ax2.set_ylabel("Total Citations", color="#2E6B3E")

            ax2_twin = ax2.twinx()
            ax2_twin.plot(yearly.index, yearly["mean"], color="#C44E52", marker="o", linewidth=2, label="Mean Citations/Doc")
            ax2_twin.set_ylabel("Mean Citations / Document", color="#C44E52")
            ax2_twin.grid(False)

            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

        fig.tight_layout()
        self._save_fig(fig, save_as)
        return fig

    def plot_top_venues(
        self, top_n: int = 12, figsize: tuple[int, int] = (10, 6), save_as: Optional[str] = None
    ) -> plt.Figure:
        """Horizontal bar chart of top publication venues / standards series."""
        venue_col = self._col("venue")
        if not venue_col:
            raise ValueError("Publication Title column not found.")

        counts = self.df[venue_col].dropna().value_counts().head(top_n)
        labels = [v[:60] + "..." if len(v) > 63 else v for v in counts.index[::-1]]

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(labels, counts.values[::-1], color=sns.color_palette("viridis", len(labels)))
        ax.set_title(f"Top {top_n} Publication Venues", fontweight="bold")
        ax.set_xlabel("Number of Documents")
        fig.tight_layout()
        self._save_fig(fig, save_as)
        return fig

    def top_cited_papers(self, top_n: int = 15) -> pd.DataFrame:
        """Return a DataFrame of the top cited documents."""
        cit_col = self._col("citations")
        if not cit_col:
            raise ValueError("Citation count column not found.")
        cols = [c for c in [self._col("title"), self._col("authors"), self._col("year"), self._col("venue"), self._col("doc_type"), cit_col] if c]
        return self.df[cols].sort_values(cit_col, ascending=False).head(top_n).reset_index(drop=True)

    def plot_top_authors(
        self, top_n: int = 15, figsize: tuple[int, int] = (12, 5), save_as: Optional[str] = None
    ) -> plt.Figure:
        """Plot top authors by document count and by total citations."""
        auth_col = self._col("authors")
        cit_col = self._col("citations")
        if not auth_col:
            raise ValueError("Authors column not found.")

        doc_counts: Counter = Counter()
        cit_counts: Counter = Counter()
        for _, row in self.df.iterrows():
            raw = row[auth_col]
            if pd.isna(raw):
                continue
            authors = [a.strip() for a in str(raw).split(";") if a.strip()]
            cits = float(row[cit_col]) if (cit_col and pd.notna(row[cit_col])) else 0.0
            for a in authors:
                doc_counts[a] += 1
                cit_counts[a] += cits

        fig, axes = plt.subplots(1, 2 if cit_col else 1, figsize=figsize)
        if not cit_col:
            axes = [axes]

        top_prod = doc_counts.most_common(top_n)
        if top_prod:
            lbls, vals = zip(*top_prod[::-1])
            axes[0].barh(lbls, vals, color="#4C72B0")
            axes[0].set_title(f"Top {top_n} Authors by Publications", fontweight="bold")
            axes[0].set_xlabel("Documents")

        if cit_col:
            top_cit = cit_counts.most_common(top_n)
            if top_cit:
                lbls, vals = zip(*top_cit[::-1])
                axes[1].barh(lbls, vals, color="#DD8452")
                axes[1].set_title(f"Top {top_n} Authors by Total Citations", fontweight="bold")
                axes[1].set_xlabel("Total Citations")

        fig.tight_layout()
        self._save_fig(fig, save_as)
        return fig

    # ------------------------------------------------------------------
    # 2. KEYWORD TABLES, 3-YEAR-GROUP EVOLUTION & VOSVIEWER OVERLAY
    # ------------------------------------------------------------------
    def _parse_semicolon_list(self, col_name: str, lowercase: bool = False) -> list[list[str]]:
        results = []
        for val in self.df[col_name]:
            if pd.isna(val):
                results.append([])
                continue
            items = []
            for t in str(val).split(";"):
                t = re.sub(r"^keywords:\s*", "", t.strip(), flags=re.IGNORECASE)
                if lowercase:
                    t = t.lower()
                if t:
                    items.append(t)
            results.append(list(dict.fromkeys(items)))
        return results

    def get_top_keywords(self, col_key: str = "author_keywords", top_n: int = 15) -> list[tuple[str, int]]:
        col_name = self._col(col_key)
        if not col_name:
            return []
        per_doc = self._parse_semicolon_list(col_name, lowercase=True)
        counts = Counter(t for sub in per_doc for t in sub)
        return counts.most_common(top_n)

    def top_keywords_table(self, col_key: str = "author_keywords", top_n: int = 20) -> pd.DataFrame:
        """Table of top terms with occurrences, share %, avg publication year, and avg citations."""
        col_name = self._col(col_key)
        if not col_name:
            raise ValueError(f"Column for '{col_key}' not found.")

        year_col = self._col("year")
        cit_col = self._col("citations")
        per_doc = self._parse_semicolon_list(col_name, lowercase=True)

        counts: Counter = Counter()
        years_acc: dict[str, list[float]] = defaultdict(list)
        cits_acc: dict[str, list[float]] = defaultdict(list)

        for idx, terms in enumerate(per_doc):
            yr = self.df.iloc[idx][year_col] if year_col else np.nan
            ct = self.df.iloc[idx][cit_col] if cit_col else np.nan
            for t in terms:
                counts[t] += 1
                if pd.notna(yr):
                    years_acc[t].append(float(yr))
                if pd.notna(ct):
                    cits_acc[t].append(float(ct))

        total_docs = len(self.df)
        rows = []
        for term, cnt in counts.most_common(top_n):
            rows.append({
                "Term": term,
                "Occurrences": cnt,
                "Docs (%)": round(cnt / total_docs * 100, 2),
                "Avg. Year": round(float(np.mean(years_acc[term])), 1) if years_acc[term] else None,
                "Avg. Citations": round(float(np.mean(cits_acc[term])), 2) if cits_acc[term] else 0.0,
            })
        return pd.DataFrame(rows)

    def top_cooccurring_pairs(
        self,
        col_key: str = "author_keywords",
        min_cooccurrences: int = 2,
        top_n: int = 20,
        lowercase: bool = True,
    ) -> pd.DataFrame:
        """Table of top co-occurring pairs (replaces clustered network graphs)."""
        col_name = self._col(col_key)
        if not col_name:
            raise ValueError(f"Column for '{col_key}' not found.")

        year_col = self._col("year")
        per_doc = self._parse_semicolon_list(col_name, lowercase=lowercase)
        pair_counts: Counter = Counter()
        pair_years: dict[tuple[str, str], list[float]] = defaultdict(list)

        for idx, terms in enumerate(per_doc):
            yr = self.df.iloc[idx][year_col] if year_col else np.nan
            for u, v in combinations(sorted(terms), 2):
                pair_counts[(u, v)] += 1
                if pd.notna(yr):
                    pair_years[(u, v)].append(float(yr))

        rows = []
        for (u, v), cnt in pair_counts.most_common(top_n):
            if cnt < min_cooccurrences:
                break
            rows.append({
                "Item 1": u,
                "Item 2": v,
                "Co-occurrences": cnt,
                "Avg. Year": round(float(np.mean(pair_years[(u, v)])), 1) if pair_years[(u, v)] else None,
            })
        return pd.DataFrame(rows)

    def plot_keyword_comparison(
        self, top_n: int = 12, figsize: tuple[int, int] = (15, 5), save_as: Optional[str] = None
    ) -> plt.Figure:
        """Side-by-side bar charts of Author Keywords, IEEE Terms, and MeSH Terms."""
        panels = [
            ("author_keywords", "Author Keywords", "#4C72B0"),
            ("ieee_terms", "IEEE Terms", "#55A868"),
            ("mesh_terms", "MeSH Terms", "#8172B3"),
        ]
        available = [(k, title, color) for k, title, color in panels if self._col(k)]
        fig, axes = plt.subplots(1, len(available), figsize=figsize)
        if len(available) == 1:
            axes = [axes]

        for ax, (k, title, color) in zip(axes, available):
            top = self.get_top_keywords(k, top_n=top_n)
            if top:
                lbls, vals = zip(*top[::-1])
                ax.barh(lbls, vals, color=color)
            ax.set_title(f"Top {top_n} {title}", fontweight="bold")
            ax.set_xlabel("Occurrences")

        fig.tight_layout()
        self._save_fig(fig, save_as)
        return fig

    def _auto_year_groups(self) -> list[tuple[int, int]]:
        """Split the dataset into 3 balanced publication year periods."""
        year_col = self._col("year")
        if not year_col:
            raise ValueError("Publication Year column not found.")
        years = self.df[year_col].dropna().astype(int)
        y_min, y_max = int(years.min()), int(years.max())
        q1 = int(np.floor(years.quantile(1 / 3)))
        q2 = int(np.floor(years.quantile(2 / 3)))
        q1 = max(y_min, min(q1, y_max - 2))
        q2 = max(q1 + 1, min(q2, y_max - 1))
        return [(y_min, q1), (q1 + 1, q2), (q2 + 1, y_max)]

    def term_counts_by_year_groups(
        self,
        source: str = "author_keywords",
        year_groups: Optional[list[tuple[int, int]]] = None,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Compare term counts across 3 groups of publication years to see where research is going.
        `source` can be 'author_keywords', 'ieee_terms', 'mesh_terms', or 'corpus' (title+abstract).
        """
        year_col = self._col("year")
        if not year_col:
            raise ValueError("Publication Year column not found.")

        groups = year_groups or self._auto_year_groups()
        if len(groups) != 3:
            raise ValueError("year_groups must contain exactly 3 (start_year, end_year) tuples.")

        if source == "corpus":
            per_doc = [set(self._tokenize(txt)) for txt in self.df["_corpus_text"]]
        else:
            col_name = self._col(source)
            if not col_name:
                raise ValueError(f"Column for '{source}' not found.")
            per_doc = [set(t) for t in self._parse_semicolon_list(col_name, lowercase=True)]

        group_counters: list[Counter] = [Counter(), Counter(), Counter()]
        group_doc_totals: list[int] = [0, 0, 0]
        overall: Counter = Counter()

        for idx, yr in enumerate(self.df[year_col]):
            if pd.isna(yr):
                continue
            y = int(yr)
            terms = per_doc[idx]
            for g_idx, (y_start, y_end) in enumerate(groups):
                if y_start <= y <= y_end:
                    group_doc_totals[g_idx] += 1
                    group_counters[g_idx].update(terms)
                    overall.update(terms)
                    break

        col_labels = [
            f"{y_start}-{y_end} (n={group_doc_totals[i]})"
            for i, (y_start, y_end) in enumerate(groups)
        ]

        rows = []
        for term, total_cnt in overall.most_common(top_n):
            c0 = group_counters[0][term]
            c1 = group_counters[1][term]
            c2 = group_counters[2][term]
            # Share % in first vs last period to highlight emerging vs declining topics
            s0 = (c0 / max(1, group_doc_totals[0])) * 100
            s2 = (c2 / max(1, group_doc_totals[2])) * 100
            diff_pp = round(s2 - s0, 2)
            rows.append({
                "Term": term,
                col_labels[0]: c0,
                col_labels[1]: c1,
                col_labels[2]: c2,
                "Total": total_cnt,
                "Share Shift (pp)": f"{diff_pp:+.2f}",
            })
        return pd.DataFrame(rows)

    def plot_term_evolution_by_year_groups(
        self,
        source: str = "author_keywords",
        year_groups: Optional[list[tuple[int, int]]] = None,
        top_n: int = 15,
        figsize: tuple[int, int] = (10, 7),
        save_as: Optional[str] = None,
    ) -> plt.Figure:
        """Heatmap of top term counts across the 3 publication year groups."""
        tbl = self.term_counts_by_year_groups(source=source, year_groups=year_groups, top_n=top_n)
        period_cols = tbl.columns[1:4]
        mat = tbl.set_index("Term")[period_cols]

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(mat, annot=True, fmt="d", cmap="YlGnBu", linewidths=0.5, ax=ax)
        ax.set_title(f"Top {top_n} Terms Across 3 Year Periods ({source})", fontweight="bold")
        ax.set_xlabel("Publication Period")
        ax.set_ylabel("")
        fig.tight_layout()
        self._save_fig(fig, save_as)
        return fig

    def build_cooccurrence_network(
        self,
        col_key: str = "author_keywords",
        min_occurrences: int = 5,
        top_n_nodes: int = 60,
        min_edge_weight: int = 2,
        lowercase: bool = True,
    ) -> nx.Graph:
        """Build a co-occurrence network with avg_year & avg_citations for Overlay Visualization."""
        col_name = self._col(col_key)
        if not col_name:
            raise ValueError(f"Column for '{col_key}' not found in dataset.")

        year_col = self._col("year")
        cit_col = self._col("citations")

        per_doc = self._parse_semicolon_list(col_name, lowercase=lowercase)
        counts: Counter = Counter()
        years_acc: dict[str, list[float]] = defaultdict(list)
        cits_acc: dict[str, list[float]] = defaultdict(list)

        for idx, terms in enumerate(per_doc):
            yr = self.df.iloc[idx][year_col] if year_col else np.nan
            ct = self.df.iloc[idx][cit_col] if cit_col else np.nan
            for t in terms:
                counts[t] += 1
                if pd.notna(yr):
                    years_acc[t].append(float(yr))
                if pd.notna(ct):
                    cits_acc[t].append(float(ct))

        eligible = {t for t, c in counts.most_common(top_n_nodes) if c >= min_occurrences}
        edge_weights: Counter = Counter()
        for terms in per_doc:
            filtered = sorted(t for t in terms if t in eligible)
            for u, v in combinations(filtered, 2):
                edge_weights[(u, v)] += 1

        G = nx.Graph()
        for t in eligible:
            G.add_node(
                t,
                count=counts[t],
                avg_year=float(np.mean(years_acc[t])) if years_acc[t] else np.nan,
                avg_citations=float(np.mean(cits_acc[t])) if cits_acc[t] else 0.0,
            )

        for (u, v), w in edge_weights.items():
            if w >= min_edge_weight:
                G.add_edge(u, v, weight=w)

        G.remove_nodes_from(list(nx.isolates(G)))
        return G

    def plot_vosviewer_overlay(
        self,
        G: nx.Graph,
        color_by: str = "avg_year",
        title: Optional[str] = None,
        figsize: tuple[int, int] = (13, 9),
        seed: int = 42,
        max_labels: int = 45,
        save_as: Optional[str] = None,
    ) -> plt.Figure:
        """Render a VOSviewer-style Overlay Visualization colored by avg_year or avg_citations."""
        if G.number_of_nodes() == 0:
            raise ValueError("Network has 0 nodes after filtering.")

        fig, ax = plt.subplots(figsize=figsize)
        pos = nx.spring_layout(G, k=1.4 / np.sqrt(G.number_of_nodes()), iterations=120, weight="weight", seed=seed)

        counts = np.array([G.nodes[n]["count"] for n in G.nodes()])
        node_sizes = 150 + 1800 * (counts - counts.min()) / max(1, counts.max() - counts.min())

        weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
        max_w = max(1, weights.max()) if len(weights) else 1
        edge_widths = 0.4 + 3.0 * (weights / max_w)
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, alpha=0.22, edge_color="#7f8c8d")

        attr_vals = np.array([G.nodes[n].get(color_by, np.nan) for n in G.nodes()], dtype=float)
        valid = attr_vals[~np.isnan(attr_vals)]
        vmin = float(np.percentile(valid, 5)) if len(valid) else 0.0
        vmax = float(np.percentile(valid, 95)) if len(valid) else 1.0
        cmap = plt.cm.viridis if color_by == "avg_year" else plt.cm.plasma
        nodes_drawn = nx.draw_networkx_nodes(
            G, pos, ax=ax, node_size=node_sizes, node_color=attr_vals,
            cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.9, edgecolors="white", linewidths=1.2
        )
        cbar_label = "Avg. Publication Year" if color_by == "avg_year" else "Avg. Citations"
        fig.colorbar(nodes_drawn, ax=ax, shrink=0.7, label=cbar_label)

        top_nodes = sorted(G.nodes(), key=lambda n: G.nodes[n]["count"], reverse=True)[:max_labels]
        for n in top_nodes:
            x, y = pos[n]
            rel = (G.nodes[n]["count"] - counts.min()) / max(1, counts.max() - counts.min())
            fontsize = 7.5 + 4.5 * rel
            ax.text(
                x, y, n, fontsize=fontsize, fontweight="bold" if rel > 0.35 else "normal",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.65),
            )

        ax.set_title(title or f"VOSviewer Overlay Visualization ({color_by})", fontsize=13, fontweight="bold")
        ax.axis("off")
        fig.tight_layout()
        self._save_fig(fig, save_as)
        return fig

    # ------------------------------------------------------------------
    # 3. VOYANT TOOLS-STYLE CORPUS & TEXT ANALYSIS (TITLE + ABSTRACT)
    # ------------------------------------------------------------------
    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"\b[a-z][a-z0-9\-]{1,}\b", str(text).lower())
        return [t for t in tokens if t not in self.stopwords and not t.isdigit()]

    def corpus_summary(self) -> pd.DataFrame:
        """Voyant 'Summary' panel returned as a clean table."""
        texts = self.df["_corpus_text"].loc[self.df["_corpus_text"].str.len() > 0]
        all_tokens: list[str] = []
        doc_lengths: list[int] = []
        for txt in texts:
            toks = self._tokenize(txt)
            all_tokens.extend(toks)
            doc_lengths.append(len(toks))

        total_tokens = len(all_tokens)
        unique_tokens = len(set(all_tokens))
        rows = [
            ("Documents with Text", len(texts)),
            ("Total Content Words", total_tokens),
            ("Unique Word Forms", unique_tokens),
            ("Vocabulary Density (TTR)", round(unique_tokens / max(1, total_tokens), 4)),
            ("Avg. Words per Document", round(float(np.mean(doc_lengths)), 1) if doc_lengths else 0.0),
        ]
        return pd.DataFrame(rows, columns=["Corpus Metric", "Value"])

    def plot_cirrus(
        self, max_words: int = 100, figsize: tuple[int, int] = (12, 6), save_as: Optional[str] = None
    ) -> plt.Figure:
        """Voyant 'Cirrus' word cloud built from Title + Abstract content words."""
        freqs: Counter = Counter()
        for txt in self.df["_corpus_text"]:
            freqs.update(self._tokenize(txt))

        wc = WordCloud(
            width=1200, height=600, background_color="white",
            max_words=max_words, colormap="viridis", prefer_horizontal=0.8
        ).generate_from_frequencies(freqs)

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Voyant Cirrus — Corpus Word Cloud (Titles & Abstracts)", fontsize=13, fontweight="bold")
        fig.tight_layout()
        self._save_fig(fig, save_as)
        return fig

    def plot_term_trends(
        self,
        terms: Optional[list[str]] = None,
        top_n: int = 6,
        relative: bool = True,
        min_docs_per_year: int = 5,
        figsize: tuple[int, int] = (12, 5),
        save_as: Optional[str] = None,
    ) -> plt.Figure:
        """Voyant 'Trends' tool: frequency of key terms over publication years."""
        year_col = self._col("year")
        if not year_col:
            raise ValueError("Publication Year column not found.")

        if terms is None:
            overall = Counter()
            for txt in self.df["_corpus_text"]:
                overall.update(self._tokenize(txt))
            terms = [w for w, _ in overall.most_common(top_n)]
        else:
            terms = [t.lower().strip() for t in terms]

        rows = []
        for yr, grp in self.df.dropna(subset=[year_col]).groupby(year_col):
            if len(grp) < min_docs_per_year:
                continue
            tokens_yr = []
            for txt in grp["_corpus_text"]:
                tokens_yr.extend(self._tokenize(txt))
            total_yr = max(1, len(tokens_yr))
            c = Counter(tokens_yr)
            rec = {"year": int(yr)}
            for t in terms:
                rec[t] = (c[t] / total_yr * 1000.0) if relative else c[t]
            rows.append(rec)

        trend_df = pd.DataFrame(rows).sort_values("year")
        fig, ax = plt.subplots(figsize=figsize)
        for t in terms:
            if t in trend_df.columns:
                ax.plot(trend_df["year"], trend_df[t], marker="o", linewidth=2, label=t)

        ax.set_title("Voyant Trends — Term Frequency Over Time", fontweight="bold")
        ax.set_xlabel("Publication Year")
        ax.set_ylabel("Relative Frequency (per 1,000 words)" if relative else "Raw Word Count")
        ax.legend(title="Terms", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        self._save_fig(fig, save_as)
        return fig

    def kwic(self, keyword: str, window: int = 6, max_rows: int = 20) -> pd.DataFrame:
        """Voyant 'Contexts' (KWIC - Keywords in Context) concordance table."""
        kw = keyword.lower().strip()
        title_col = self._col("title")
        year_col = self._col("year")
        records = []

        for _, row in self.df.iterrows():
            words = re.findall(r"\b[\w\-]+\b", str(row["_corpus_text"]))
            for i, w in enumerate(words):
                if w.lower() == kw:
                    left = " ".join(words[max(0, i - window):i])
                    right = " ".join(words[i + 1:i + 1 + window])
                    records.append({
                        "Year": int(row[year_col]) if (year_col and pd.notna(row[year_col])) else None,
                        "Document Title": (str(row[title_col])[:55] + "...") if title_col else "",
                        "Left Context": left,
                        "Keyword": w,
                        "Right Context": right,
                    })
                    if len(records) >= max_rows:
                        return pd.DataFrame(records)
        return pd.DataFrame(records)

    def collocates(self, keyword: str, window: int = 5, top_n: int = 15) -> pd.DataFrame:
        """Voyant 'Collocates' tool: words most frequently appearing near `keyword`."""
        kw = keyword.lower().strip()
        colloc_counts: Counter = Counter()
        for txt in self.df["_corpus_text"]:
            toks = self._tokenize(txt)
            for i, w in enumerate(toks):
                if w == kw:
                    ctx = toks[max(0, i - window):i] + toks[i + 1:i + 1 + window]
                    for c in ctx:
                        if c != kw:
                            colloc_counts[c] += 1
        return pd.DataFrame(colloc_counts.most_common(top_n), columns=["Collocate", "Co-occurrences"])

    def plot_ngrams(
        self,
        ngram_range: tuple[int, int] = (2, 2),
        top_n: int = 15,
        min_df: int = 3,
        figsize: tuple[int, int] = (10, 6),
        save_as: Optional[str] = None,
    ) -> plt.Figure:
        """Plot top n-grams (bigrams/trigrams) in the corpus using scikit-learn."""
        texts = self.df["_corpus_text"].loc[self.df["_corpus_text"].str.len() > 0]
        clean_stops = sorted({w for w in self.stopwords if "'" not in w})
        vec = CountVectorizer(
            ngram_range=ngram_range,
            stop_words=clean_stops,
            min_df=min_df,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]+\b",
        )
        X = vec.fit_transform(texts)
        sums = np.asarray(X.sum(axis=0)).ravel()
        vocab = vec.get_feature_names_out()
        top_idx = np.argsort(sums)[::-1][:top_n]

        labels = [vocab[i] for i in top_idx[::-1]]
        values = [int(sums[i]) for i in top_idx[::-1]]

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(labels, values, color="#8172B3")
        n_lbl = "Bigrams" if ngram_range == (2, 2) else f"{ngram_range}-grams"
        ax.set_title(f"Top {top_n} {n_lbl} in Titles & Abstracts", fontweight="bold")
        ax.set_xlabel("Total Occurrences")
        fig.tight_layout()
        self._save_fig(fig, save_as)
        return fig

    def plot_distinctive_terms_by_group(
        self,
        group_col_key: str = "doc_type",
        top_n: int = 8,
        max_groups: int = 4,
        figsize: tuple[int, int] = (14, 8),
        save_as: Optional[str] = None,
    ) -> plt.Figure:
        """TF-IDF distinctive terms per top document type or group."""
        col_name = self._col(group_col_key)
        if not col_name:
            raise ValueError(f"Column for '{group_col_key}' not found.")

        top_groups = self.df[col_name].value_counts().head(max_groups).index
        sub_df = self.df[self.df[col_name].isin(top_groups)]
        grouped = sub_df.groupby(col_name)["_corpus_text"].apply(lambda s: " ".join(s)).dropna()
        clean_stops = sorted({w for w in self.stopwords if "'" not in w})
        vec = TfidfVectorizer(
            stop_words=clean_stops,
            max_features=3000,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]+\b",
        )
        tfidf = vec.fit_transform(grouped.values)
        vocab = vec.get_feature_names_out()

        n_groups = len(grouped)
        ncols = min(2, n_groups)
        nrows = int(np.ceil(n_groups / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes_flat = np.atleast_1d(axes).ravel()

        for idx, grp_name in enumerate(grouped.index):
            ax = axes_flat[idx]
            scores = tfidf[idx].toarray().ravel()
            top_idx = np.argsort(scores)[::-1][:top_n]
            lbls = [vocab[i] for i in top_idx[::-1]]
            vals = [scores[i] for i in top_idx[::-1]]
            ax.barh(lbls, vals, color=sns.color_palette("Set2")[idx % 8])
            ax.set_title(f"Distinctive Terms: {grp_name}", fontweight="bold", fontsize=11)
            ax.set_xlabel("TF-IDF Score")

        for extra_ax in axes_flat[n_groups:]:
            extra_ax.set_visible(False)

        fig.tight_layout()
        self._save_fig(fig, save_as)
        return fig

