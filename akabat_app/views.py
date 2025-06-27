# akabat_app/views.py

import os
import io 
import shutil
import uuid
import zipfile
import pytz
import ast

from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
from django.http import FileResponse
from django.contrib import messages
from .forms import CSVUploadForm, ScopusSearchForm
from django.utils.safestring import mark_safe

import requests
import csv

from akabat_app.akabat_core.controller.controller import Controller

from django.http import HttpResponse
from django.utils import timezone
import pandas as pd
from plotly.utils import PlotlyJSONEncoder
import json
import plotly.io as pio
from datetime import datetime
import sqlite3

from .models import SavedProject

from django.http import HttpResponseRedirect
from django.urls import reverse
from django.views.decorators.http import require_GET
from django.http import JsonResponse


def home_view(request):
    # Clear previous session
    request.session.flush()

    # Obtain or generate user_uuid
    user_uuid = request.COOKIES.get("user_uuid")
    if not user_uuid:
        response = redirect("home")
        new_uuid = str(uuid.uuid4())
        response.set_cookie("user_uuid", new_uuid, max_age=60*60*24*365*2)  # 2 years max
        return response

    # If POST to delete project
    if request.method == "POST" and "delete_project_id" in request.POST:
        project_id = request.POST.get("delete_project_id")
        deleted = SavedProject.objects.filter(id=project_id, user_uuid=user_uuid).delete()
        if deleted[0] > 0:
            messages.success(request, "Project deleted successfully.")
        else:
            messages.error(request, "Project not found or unauthorized.")
        return redirect("home")

    # Get users projects
    projects = SavedProject.objects.filter(user_uuid=user_uuid).order_by("-last_modified")
    return render(request, "akabat_app/home.html", {
        "projects": projects
    })


def project_list_view(request):
    user_uuid = request.COOKIES.get("user_uuid")
    if not user_uuid:
        return redirect("home")

    if request.method == "POST":
        project_id = request.POST.get("project_id")
        action = request.POST.get("action")

        if action == "delete":
            SavedProject.objects.filter(id=project_id, user_uuid=user_uuid).delete()
        elif action == "rename_mode":
            request.session["edit_project_id"] = project_id
            return redirect("project_list")

        elif action == "rename":
            new_name = request.POST.get("new_name", "").strip()
            if new_name:
                SavedProject.objects.filter(id=project_id, user_uuid=user_uuid).update(name=new_name)

        elif action == "duplicate":
            original = get_object_or_404(SavedProject, id=project_id, user_uuid=user_uuid)
            SavedProject.objects.create(
                name=original.name + " (Copy)",
                user_uuid=original.user_uuid,
                folder_path=original.folder_path,
                last_step=original.last_step,
                session_csvs=original.session_csvs,
                session_excluded=original.session_excluded,
                session_included=original.session_included,
            )

        return redirect("project_list")
    edit_project_id = request.session.pop("edit_project_id", None)
    projects = SavedProject.objects.filter(user_uuid=user_uuid).order_by("-last_modified")
    return render(request, "akabat_app/project_list.html", {
        "projects": projects,
        "edit_project_id": edit_project_id,
    })


def resume_project_view(request, project_id):
    project = get_object_or_404(SavedProject, id=project_id)
    request.session["akabat_project_folder"] = project.folder_path
    request.session["csvs"] = project.session_csvs or []
    request.session["excluded_keywords"] = project.session_excluded or {}
    request.session["included_keywords"] = project.session_included or {}
    step_to_view= {
        "import_csv": "upload_csv", 
        "preferences": "preferences",
        "group_semantic": "group_semantic",
        "cluster_authors": "cluster_authors",
        "create_database": "create_db",
        "plots": "plots",
    }
    view_name = step_to_view.get(project.last_step, "home")
    return redirect(view_name)

def welcome_view(request):
    context = {
        "current_step": 1,
        "total_steps": 9,
        "step_title": "Welcome",
        "next_url": "/akabat/import/" 
    }
    return render(request, "akabat_app/wizard_welcome.html", context)


def delete_project_view(request, project_id):
    SavedProject.objects.filter(id=project_id).delete()
    return HttpResponseRedirect(reverse("home"))

def get_project_folder(request):
    pf = request.session.get("akabat_project_folder")
    if not pf:
        raise ValueError("No project folder found in session. Please start a new project.")
    return pf

def get_controller(request):
    project_folder = get_project_folder(request)
    db_path = os.path.join(project_folder, "db.sqlite3")
    return Controller(db_path=db_path)

def get_csvs_folder(request):
    """
    Obtains the route to the folder 'csvs' inside the project
    """
    return os.path.join(get_project_folder(request), "csvs")


def get_openalex_concepts(doi):
    headers = {"Accept": "application/json"}
    url = f"https://api.openalex.org/works/https://doi.org/{doi}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        concepts = data.get("concepts", [])
        return [concept["display_name"] for concept in concepts]
    return []


def save_current_project_state(request, step: str):
    user_uuid = request.COOKIES.get("user_uuid")
    project_folder = request.session.get("akabat_project_folder", "/tmp/akabat_project")

    if user_uuid:
        local_tz = pytz.timezone("Europe/Berlin")  # o Europe/Madrid
        local_now = timezone.now().astimezone(local_tz)
        name = f"Akabat Project {local_now.strftime('%Y-%m-%d %H:%M:%S')}"

        SavedProject.objects.update_or_create(
            user_uuid=user_uuid,
            folder_path=project_folder,
            defaults={
                "name": name,
                "last_step": step,
                "session_csvs": request.session.get("csvs", []),
                "session_excluded": request.session.get("excluded_keywords", {}),
                "session_included": request.session.get("included_keywords", {}),
            }
        )

        # Clean flush messages
        storage = messages.get_messages(request)
        list(storage)


def import_csv_view(request):
    current_step, total_steps = 2, 9
    csv_list = request.session.get("csvs", [])

    def save_session_and_redirect():
        request.session["csvs"] = csv_list
        return redirect("upload_csv")

    # 1. SAVE COLUMN NAMES DEFINITION
    if request.method == "POST" and "edit_id" in request.POST:
        edit_id = request.POST.get("edit_id")
        for csv_file in csv_list:
            if csv_file["id"] == edit_id:
                csv_file["preferences"] = {
                    key: request.POST.get(key, "")
                    for key in ["title", "author", "publication_year", "doi", "country", "keywords", "citations"]
                }
                break
        return save_session_and_redirect()


    # 2. DELETE FILE
    if request.method == "POST" and "remove_id" in request.POST:
        remove_id = request.POST.get("remove_id")
        csv_list = [csv_file for csv_file in csv_list if csv_file["id"] != remove_id]
        return save_session_and_redirect()

    # 3. MANUAL UPLOAD
    if request.method == "POST" and "local_upload" in request.POST:
        files = request.FILES.getlist("csv_files")
        if files:
            project_folder = request.session.get("akabat_project_folder")
            if not project_folder:
                project_id = uuid.uuid4().hex
                project_folder = os.path.join(settings.MEDIA_ROOT, "projects", project_id)
                os.makedirs(project_folder, exist_ok=True)
                request.session["akabat_project_folder"] = project_folder

            csvs_folder = os.path.join(project_folder, "csvs")
            os.makedirs(csvs_folder, exist_ok=True)

            for f in files:
                save_path = os.path.join(csvs_folder, f.name)
                with open(save_path, "wb") as dst:
                    for chunk in f.chunks():
                        dst.write(chunk)
                try:
                    pd.read_csv(save_path, nrows=1)
                except Exception as e:
                    messages.error(request, f"❌ Error while processing file '{f.name}': {e}")
                    continue

                csv_list.append({
                    "id": uuid.uuid4().hex,
                    "filename": f.name,
                    "path": save_path,
                    "source": "local",
                    "preferences": {}
                })
            return save_session_and_redirect()

    # 4. UPLOAD FROM SCOPUS
    if request.method == "POST" and "scopus_fetch" in request.POST:
        form = ScopusSearchForm(request.POST)
        if form.is_valid():
            kw_raw = form.cleaned_data["keywords"]
            kw_list = [k.strip() for k in kw_raw.split(",") if k.strip()]
            year_from = form.cleaned_data["year_from"]
            year_to = form.cleaned_data["year_to"]
            max_res = form.cleaned_data["max_results"] or 25
            author = form.cleaned_data["author"]
            language = form.cleaned_data["language"]
            order_by = form.cleaned_data["order_by"]

            query = form.cleaned_data.get("advanced_query", "").strip()
            if not query:
                kw_raw = form.cleaned_data["keywords"]
                kw_list = [k.strip() for k in kw_raw.split(",") if k.strip()]
                logic = form.cleaned_data.get("keyword_logic") or "OR"

                query_parts = []
                if kw_list:
                    kw_joined = f" {logic} ".join([f'KEY("{kw}")' for kw in kw_list])
                    query_parts.append(f"({kw_joined})")

                if year_from:
                    query_parts.append(f"PUBYEAR > {year_from - 1}")
                if year_to:
                    query_parts.append(f"PUBYEAR < {year_to + 1}")
                if author:
                    query_parts.append(f'AUTHLASTNAME("{author}")')
                if language:
                    query_parts.append(f'LANGUAGE("{language}")')

                query = " AND ".join(query_parts)

            elif not any(x in query for x in ["KEY", "TITLE", "LIMIT", "PUBYEAR"]):
                messages.error(request, "⚠️ Your advanced query may be incomplete. Please check the syntax.")
                return redirect("upload_csv")

            headers = {"X-ELS-APIKey": settings.SCOPUS_API_KEY}
            params = {"query": query, "count": max_res, "httpAccept": "application/json"}
            if order_by == "date":
                params["sort"] = "date"
            entries = []
            try:
                for start in range(0, max_res, 25):  
                    params["start"] = start
                    params["count"] = min(25, max_res - start)
                    resp = requests.get("https://api.elsevier.com/content/search/scopus", params=params, headers=headers)
                    resp.raise_for_status()
                    batch = resp.json().get("search-results", {}).get("entry", [])
                    if not batch:
                        break  # There are no more results
                    entries.extend(batch)
            except Exception as e:
                messages.error(request, f"Error while accessing Scopus API: {e}")
                return redirect("upload_csv")

            if not entries or all(not e.get("dc:title") for e in entries):
                messages.error(request, "No valid results extracted from Scopus. Verify the filters or try again later.")
                return redirect("upload_csv")

            keywords_map = {}
            citations_map = {}
            # OBTAIN KEYWORDS FROM OPENALEX
            for e in entries:
                eid = e.get("eid", "")
                doi = e.get("prism:doi", "")
                kws = []
                cited_count = e.get("citedby-count", 0)
                citations_map[eid] = cited_count

                if eid:
                    try:
                        r2 = requests.get(f"https://api.elsevier.com/content/abstract/scopus_id/{eid}",
                                        params={"httpAccept": "application/json"}, headers=headers)
                        if r2.ok:
                            j2 = r2.json().get("abstracts-retrieval-response", {})
                            for group in ["authkeywords", "indexterms"]:
                                for kw_obj in j2.get(group, {}).get("index-term" if group == "indexterms" else "author-keyword", []):
                                    text = kw_obj.get("$")
                                    if text:
                                        kws.append(text)

                    except Exception:
                        pass

                if not kws and doi:
                    kws = get_openalex_concepts(doi)
                keywords_map[eid] = list(dict.fromkeys(kws))

            project_folder = request.session.get("akabat_project_folder")
            if not project_folder:
                project_id = uuid.uuid4().hex
                project_folder = os.path.join(settings.MEDIA_ROOT, "projects", project_id)
                os.makedirs(project_folder, exist_ok=True)
                request.session["akabat_project_folder"] = project_folder

            csvs_folder = os.path.join(project_folder, "csvs")
            os.makedirs(csvs_folder, exist_ok=True)
            keywords_name = "_".join(kw_list).replace(" ", "_").lower()
            keywords_name = "".join(c for c in keywords_name if c.isalnum() or c == "_")[:50]
            filename = f"Scopus_{keywords_name or 'consulta'}.csv"
            outfn = os.path.join(csvs_folder, filename)

            try:
                with open(outfn, "w", newline="", encoding="utf-8") as fcsv:
                    writer = csv.writer(fcsv)
                    writer.writerow(["title", "author", "publication_year", "doi", "country", "keywords", "citations"])
                    for e in entries:
                        eid = e.get("eid", "")
                        cover_date = e.get("prism:coverDate", "")
                        year = ""
                        if cover_date and len(cover_date) >= 4 and cover_date[:4].isdigit():
                            year = cover_date[:4]
                        else:
                            print(f"⚠️ Invalid or missing coverDate for EID {eid}: {cover_date}")

                        row = [
                            e.get("dc:title", ""),
                            e.get("dc:creator", ""),
                            year,
                            e.get("prism:doi", ""),
                            e.get("affiliation", [{}])[0].get("affiliation-country", ""),
                            "; ".join(kw.replace(";", "").strip() for kw in keywords_map.get(eid, [])),
                            citations_map.get(eid, 0)
                        ]
                        writer.writerow(row)

            except Exception as e:
                messages.error(request, f"Error while saving CSV file: {e}")
                return redirect("upload_csv")

            csv_list.append({
                "id": uuid.uuid4().hex,
                "filename": os.path.basename(outfn),
                "path": outfn,
                "source": "scopus",
                "preferences": {
                    "title": "title",
                    "author": "author",
                    "publication_year": "publication_year",
                    "doi": "doi",
                    "country": "country",
                    "keywords": "keywords",
                    "citations": "citations"
                }
            })

            return save_session_and_redirect()

     # 5.  SAVE PROJECT
    if request.method == "POST" and "save_project" in request.POST:
        save_current_project_state(request, step="import_csv")
        return redirect("home")

    if request.method == "POST" and request.POST.get("finalize") == "1":
        # Save disabled articles
        disabled_raw = request.POST.get("disabled_rows", "")
        disabled_indices = set(int(x) for x in disabled_raw.split(",") if x.isdigit())
        request.session["disabled_indices"] = sorted(disabled_indices)
        return redirect("review_import")


    context = {
        "current_step": current_step,
        "total_steps": total_steps,
        "step_title": "Import CSVs",
        "form": CSVUploadForm(),
        "scopus_form": ScopusSearchForm(),
        "csv_files": csv_list,
        "column_fields": ["title", "author", "publication_year", "doi", "country", "keywords", "citations"],
        "required_fields": ["title", "author", "publication_year", "doi", "keywords"],

    }
    return render(request, "akabat_app/import_csv.html", context)

def review_import_view(request):
    csv_list = request.session.get("csvs", [])
    if not csv_list:
        return redirect("upload_csv")
    required_keys = ["title", "author", "publication_year", "doi", "country", "keywords", "citations"]
    for csv_file in csv_list:
        if csv_file["source"] == "local":
            if not all(csv_file["preferences"].get(k) for k in ["title", "author", "publication_year", "doi", "keywords"]):
                return redirect("upload_csv")

    temp_merge_dir = os.path.join(settings.MEDIA_ROOT, "temp_merge_csvs")
    shutil.rmtree(temp_merge_dir, ignore_errors=True)
    os.makedirs(temp_merge_dir, exist_ok=True)

    all_dfs = []
    for csv_file in csv_list:
        df = pd.read_csv(csv_file["path"])
        prefs = csv_file["preferences"]
        rename_map = {
            prefs[k].strip(): k
            for k in required_keys
            if prefs.get(k) and prefs[k].strip() in df.columns
        }
        df["__source_file__"] = csv_file["filename"]
        df = df.rename(columns=rename_map)
        if csv_file["source"] == "local":
            csv_file["preferences"] = {key: key for key in required_keys}
        out_path = os.path.join(temp_merge_dir, csv_file["filename"])
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        all_dfs.append(df)

    controller = get_controller(request)
    controller._preferences.csv_import_column_names = {k: k for k in required_keys}
    controller._preferences.csv_import_column_mappings_by_file = {
        csv["filename"]: {v: k for k, v in csv["preferences"].items()}
        for csv in csv_list
    }
    controller._preferences.excluded_keywords_at_csv_import = []
    controller._preferences.excluded_starting_by_keywords_at_csv_import = []
    controller._preferences.excluded_contains_keywords_at_csv_import = []
    disabled_indices = set(request.session.get("disabled_indices", []))
    removed = controller.import_all_csvs(temp_merge_dir, disabled_indices=disabled_indices)
    controller._removed_duplicates = removed

    df = controller._raw_papers.copy().reset_index(drop=True)
    request.session["last_raw_df"] = df.to_dict("records") 


    csv_files_by_name = {f["filename"]: idx for idx, f in enumerate(csv_list)}

    preview_rows = []
    for i, row in df.iterrows():
        source_file = os.path.basename(row.get("__source_file__", ""))
        row_dict = {
            "values": list(row.values),
            "source_id": csv_files_by_name.get(source_file, 0),
            "is_disabled": i in disabled_indices,
        }
        preview_rows.append(row_dict)

    file_legend = [{"name": f["filename"]} for f in csv_list]
    request.session["last_raw_df"] = controller._raw_papers.to_dict(orient="records")

    return render(request, "akabat_app/review_import.html", {
        "current_step": 2,
        "total_steps": 9,
        "step_title": "Review Imported Data",
        "removed_count": removed,
        "preview_rows": preview_rows,
        "column_headers": list(df.columns),
        "file_legend": file_legend,
        "total_articles": len(df),
        "next_url": "/akabat/preferences/",
        "disabled_indices": list(disabled_indices),
    })

def preferences_view(request):
    current_step, total_steps = 3, 9
    csvs_folder = os.path.join(settings.MEDIA_ROOT, "temp_merge_csvs")
    controller = get_controller(request)
    controller._preferences.csv_import_column_mappings_by_file = {
        csv["filename"]: {
            k: k 
            for k in ["title", "author", "publication_year", "doi", "country", "keywords", "citations"]
        }
        for csv in request.session.get("csvs", [])
    }
    disabled_indices = set(request.session.get("disabled_indices", []))
    controller.import_all_csvs(csvs_folder)
    controller._raw_papers = controller._raw_papers.reset_index(drop=True)
    controller._raw_papers["__row_index__"] = controller._raw_papers.index
    controller._raw_papers = controller._raw_papers[~controller._raw_papers["__row_index__"].isin(disabled_indices)]


    if request.method == "POST":
        action = request.POST.get("action")
        # Save blacklist
        request.session["excluded_keywords"] = {
            "excluded_starting_by_keywords_at_csv_import":
                request.POST.getlist("excluded_starting_by_keywords_at_csv_import[]"),
            "excluded_keywords_at_csv_import":
                request.POST.getlist("excluded_keywords_at_csv_import[]"),
            "excluded_contains_keywords_at_csv_import":
                request.POST.getlist("excluded_contains_keywords_at_csv_import[]"),
            "excluded_keywords_in_plot": [],
        }
        # Save whitelist
        request.session["included_keywords"] = {
            "included_starting_by_keywords":
                request.POST.getlist("included_starting_by_keywords[]"),
            "included_keywords":
                request.POST.getlist("included_keywords[]"),
            "included_contains_keywords":
                request.POST.getlist("included_contains_keywords[]"),
        }

        if action == "save":
            save_current_project_state(request, step="preferences")
            return redirect("home")
        elif action == "next":
            return redirect("/akabat/process_keywords/")

    default_excluded_keywords = request.session.get("excluded_keywords", {
        "excluded_starting_by_keywords_at_csv_import": [],
        "excluded_keywords_at_csv_import": [],
        "excluded_contains_keywords_at_csv_import": [],
        "excluded_keywords_in_plot": [],
    })

    default_included_keywords = request.session.get("included_keywords", {
        "included_starting_by_keywords": [],
        "included_keywords": [],
        "included_contains_keywords": [],
    })

    unique_keywords = sorted(set(
        kw.strip()
        for row in controller._raw_papers["keywords"]
        if isinstance(row, list)
        for kw in row
    ))


    return render(request, "akabat_app/wizard_preferences.html", {
        "current_step": current_step,
        "total_steps": total_steps,
        "unique_keywords": unique_keywords,
        "step_title": "Configure Preferences",
        "default_excluded_keywords": default_excluded_keywords,
        "default_included_keywords": default_included_keywords,
    })

def process_keywords_view(request):
    current_step, total_steps = 4, 9
    csvs_folder = os.path.join(settings.MEDIA_ROOT, "temp_merge_csvs")
    controller = get_controller(request)
    controller._preferences.csv_import_column_mappings_by_file = {
        csv["filename"]: {
            k: k 
            for k in ["title", "author", "publication_year", "doi", "country", "keywords", "citations"]
        }
        for csv in request.session.get("csvs", [])
    }
    controller.import_all_csvs(csvs_folder)
    included = request.session.get("included_keywords", {
        "included_starting_by_keywords": [],
        "included_keywords": [],
        "included_contains_keywords": [],
    })
    excluded = request.session.get("excluded_keywords", {
        "excluded_starting_by_keywords_at_csv_import": [],
        "excluded_keywords_at_csv_import": [],
        "excluded_contains_keywords_at_csv_import": [],
        "excluded_keywords_in_plot": [],
    })
    controller._preferences.excluded_keywords_at_csv_import = excluded["excluded_keywords_at_csv_import"]
    controller._preferences.excluded_starting_by_keywords_at_csv_import = excluded["excluded_starting_by_keywords_at_csv_import"]
    controller._preferences.excluded_contains_keywords_at_csv_import = excluded.get("excluded_contains_keywords_at_csv_import", [])
    controller._raw_papers = controller._paper_loader.apply_keyword_exclusions(
        controller._raw_papers,
        included_keywords=included["included_keywords"],
        included_starting_by_keywords=included["included_starting_by_keywords"],
        included_contains_keywords=included["included_contains_keywords"],
        excluded_keywords=excluded["excluded_keywords_at_csv_import"],
        excluded_starting_by_keywords=excluded["excluded_starting_by_keywords_at_csv_import"],
        excluded_contains_keywords=excluded["excluded_contains_keywords_at_csv_import"]
        )

    controller.generate_unique_keywords()
    unique = controller._data.unique_keywords
    project_folder = request.session["akabat_project_folder"]
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    with open(os.path.join(project_folder, "unique_keywords.json"), "w", encoding="utf-8") as f:
        json.dump(list(unique), f, indent=2, ensure_ascii=False)

    unique = sorted(list(unique), key=str.lower)
    return render(request, "akabat_app/wizard_keywords.html", {
        "current_step": current_step,
        "total_steps": total_steps,
        "step_title": "Generate Keywords",
        "result_message": f"Number of unique keywords: {len(unique)}",
        "unique_keywords": list(unique),
        "next_url": "/akabat/group_semantic/",
        "back_url": "/akabat/preferences/",
    })


def preview_raw_papers_view(request):
    current_step, total_steps = 3, 9
    csvs_folder = get_csvs_folder(request)
    dfs = []
    for fname in sorted(os.listdir(csvs_folder)):
        if fname.lower().endswith(".csv"):
            full = os.path.join(csvs_folder, fname)
            try:
                df = pd.read_csv(full, encoding="utf-8-sig")
                df["__source_file__"] = fname  
                dfs.append(df)
            except Exception as e:
                dfs.append(pd.DataFrame({
                    "__error__": [f"Error leyendo {fname}: {e}"]
                }))
    if dfs:
        raw_all = pd.concat(dfs, ignore_index=True, sort=False)
    else:
        raw_all = pd.DataFrame([{"__warning__": "NO CSV FOUND."}])

    html_table = raw_all.to_html(
        index=False,
        classes="table table-sm table-bordered",
        table_id="preview-raw-table",
        justify="center",
        escape=False
    )

    return render(request, "akabat_app/preview_raw_papers.html", {
        "current_step": current_step,
        "total_steps": total_steps,
        "step_title": "Preview Raw Papers",
        "table_html": mark_safe(html_table),
    })


def group_semantic_view(request):
    current_step, total_steps = 5, 9
    controller = get_controller(request)
    uniq_path = os.path.join(get_project_folder(request), "unique_keywords.json")

    with open(uniq_path, "r", encoding="utf-8") as f:
        controller._data.unique_keywords = json.load(f)

    groups = None
    silhouette_score_used = None
    selected_mode = "auto"
    selected_k = 10
    selected_linkage = "ward"
    selected_affinity = "euclidean"
    distance_threshold = None  

    if request.method == "POST":
        action = request.POST.get("action")
        if action == "cluster":
            # User has clicked "Group Keywords"
            distance_threshold = request.POST.get("distance_threshold")
            selected_mode = request.POST.get("clustering_mode", "auto")
            selected_linkage = request.POST.get("linkage", "ward")
            selected_affinity = request.POST.get("affinity", "euclidean")
            if selected_mode == "manual":
                selected_k = int(request.POST.get("num_clusters", 10))
                groups, silhouette_score_used = controller.group_keywords_by_semantic_similarity(
                    k=selected_k,
                    linkage=selected_linkage,
                    affinity=selected_affinity,
                    distance_threshold=float(distance_threshold) if distance_threshold else None
                )
            else:
                groups, silhouette_score_used = controller.group_keywords_by_semantic_similarity(
                    linkage=selected_linkage,
                    affinity=selected_affinity,
                    distance_threshold=float(distance_threshold) if distance_threshold else None
                )
            controller._data.unique_keywords_groups = groups

        elif action == "save_and_continue":
            # User clicked Next
            group_names = request.POST.getlist("group_names[]")
            keywords_by_group = request.POST.getlist("group_keywords[]")
            group_disabled = request.POST.getlist("group_disabled[]")
            parsed_groups = {}
            for name, kws_str, disabled in zip(group_names, keywords_by_group, group_disabled):
                if disabled == "1":
                    continue  # Discard disabled group
                try:
                    parsed_groups[name] = {
                        "keywords": ast.literal_eval(kws_str),
                        "enabled": disabled != "1"
                    }
                except Exception:
                    pass


            controller._data.unique_keywords_groups = parsed_groups
            groups_json = os.path.join(get_project_folder(request), "keyword_groups.json")
            with open(groups_json, "w", encoding="utf-8") as f:
                json.dump(parsed_groups, f, indent=2, ensure_ascii=False)
            return redirect("/akabat/cluster_authors/")
        
        elif action == "save":
            group_names = request.POST.getlist("group_names[]")
            keywords_by_group = request.POST.getlist("group_keywords[]")
            group_disabled = request.POST.getlist("group_disabled[]")
            parsed_groups = {}
            for name, kws_str, disabled in zip(group_names, keywords_by_group, group_disabled):
                try:
                    if disabled != "1":
                        parsed_groups[name] = ast.literal_eval(kws_str)
                except Exception:
                    pass

            controller._data.unique_keywords_groups = parsed_groups
            groups_json = os.path.join(get_project_folder(request), "keyword_groups.json")
            with open(groups_json, "w", encoding="utf-8") as f:
                json.dump(parsed_groups, f, indent=2, ensure_ascii=False)
            save_current_project_state(request, step="group_semantic")
            return redirect("home")


    else:
        groups = controller._data.unique_keywords_groups or {}
        if groups and isinstance(list(groups.values())[0], list):
            groups = {
                name: {"keywords": kws, "enabled": True}
                for name, kws in groups.items()
            }
        elif groups and isinstance(list(groups.values())[0], dict):
            for name in groups:
                data = groups[name]
                if not isinstance(data, dict):
                    groups[name] = {"keywords": [], "enabled": True}
                else:
                    if "keywords" not in data:
                        data["keywords"] = []
                    if "enabled" not in data:
                        data["enabled"] = True
        controller._data.unique_keywords_groups = groups  

    max_clusters = max(2, len(controller._data.unique_keywords) - 1)
    return render(request, "akabat_app/wizard_groups.html", {
        "current_step": current_step,
        "total_steps": total_steps,
        "step_title": "Group Semantically",
        "groups": groups,
        "max_clusters": max_clusters,
        "distance_threshold": distance_threshold,
        "silhouette_score_used": silhouette_score_used,
        "selected_mode": selected_mode,
        "selected_k": selected_k,
        "selected_linkage": selected_linkage,
        "selected_affinity": selected_affinity,
        "next_url": "/akabat/cluster_authors/",
    })



def cluster_authors_view(request):
    current_step, total_steps = 6, 9
    controller = get_controller(request)
    csvs_folder = os.path.join(settings.MEDIA_ROOT, "temp_merge_csvs")
    controller._preferences.csv_import_column_mappings_by_file = {
        csv["filename"]: {k: k for k in ["title", "author", "publication_year", "doi", "country", "keywords", "citations"]}
        for csv in request.session.get("csvs", [])
    }
    controller.import_all_csvs(csvs_folder)
    included = request.session.get("included_keywords", {
        "included_starting_by_keywords": [],
        "included_keywords": [],
        "included_contains_keywords": [],
    })
    excluded = request.session.get("excluded_keywords", {
        "excluded_starting_by_keywords_at_csv_import": [],
        "excluded_keywords_at_csv_import": [],
        "excluded_contains_keywords_at_csv_import": [],
    })
    controller._preferences.excluded_keywords_at_csv_import = excluded["excluded_keywords_at_csv_import"]
    controller._preferences.excluded_starting_by_keywords_at_csv_import = excluded["excluded_starting_by_keywords_at_csv_import"]
    controller._preferences.excluded_contains_keywords_at_csv_import = excluded.get("excluded_contains_keywords_at_csv_import", [])
    controller._raw_papers = controller._paper_loader.apply_keyword_exclusions(
        controller._raw_papers,
        included_keywords=included["included_keywords"],
        included_starting_by_keywords=included["included_starting_by_keywords"],
        included_contains_keywords=included["included_contains_keywords"],
        excluded_keywords=excluded["excluded_keywords_at_csv_import"],
        excluded_starting_by_keywords=excluded["excluded_starting_by_keywords_at_csv_import"],
        excluded_contains_keywords=excluded["excluded_contains_keywords_at_csv_import"]
    )

    clusters = {}
    if request.method == "POST":
        action = request.POST.get("action")
        if action == "cluster":
            mode = request.POST.get("clustering_mode", "auto")
            k_str = request.POST.get("num_clusters")
            k = int(k_str) if mode == "manual" and k_str else None
            threshold_str = request.POST.get("stopword_threshold")
            threshold = float(threshold_str) if mode == "manual" and threshold_str else 0.5
            top_n_str = request.POST.get("top_n_keywords")
            top_n = int(top_n_str) if mode == "manual" and top_n_str else 3
            distance_threshold = request.POST.get("distance_threshold")
            distance_threshold = float(distance_threshold) if distance_threshold else None
            linkage = request.POST.get("linkage", "average")
            affinity = request.POST.get("affinity", "euclidean")
            clusters, stopwords, score = controller.cluster_authors_by_keywords(
                k=k,
                stopword_threshold=threshold,
                top_n_keywords=top_n,
                distance_threshold=distance_threshold,
                linkage=linkage,
                affinity=affinity
            )
            controller._data.author_clusters = clusters
            controller._data.removed_keywords = stopwords
            controller._data.cluster_quality_score = score
            cluster_path = os.path.join(get_project_folder(request), "author_clusters.json")
            with open(cluster_path, "w", encoding="utf-8") as f:
                json.dump(clusters, f, indent=2, ensure_ascii=False)

        elif action == "save_and_continue":
            group_names = request.POST.getlist("group_names[]")
            group_authors = request.POST.getlist("group_authors[]")
            group_disabled = request.POST.getlist("group_disabled[]")

            parsed = {}
            for name, authors_str, disabled_flag in zip(group_names, group_authors, group_disabled):
                try:
                    authors = ast.literal_eval(authors_str)
                    is_disabled = disabled_flag == "1"
                    if authors:  # Only save groups that have authors
                        parsed[name] = {
                            "authors": authors,
                            "keywords": [],
                            "disabled": is_disabled
                        }
                except Exception:
                    continue

            if parsed:
                controller._data.author_clusters = parsed
                cluster_path = os.path.join(get_project_folder(request), "author_clusters.json")
                with open(cluster_path, "w", encoding="utf-8") as f:
                    json.dump(parsed, f, indent=2, ensure_ascii=False)
                backup_path = cluster_path.replace(".json", "_backup.json")
                with open(backup_path, "w", encoding="utf-8") as f:
                    json.dump(parsed, f, indent=2, ensure_ascii=False)
            return redirect("/akabat/create_db/")
    else:
        clusters = controller._data.author_clusters

    return render(request, "akabat_app/wizard_authors.html", {
        "current_step": current_step,
        "total_steps": total_steps,
        "step_title": "Cluster Authors",
        "clusters": clusters,
        "stopwords": list(getattr(controller._data, "removed_keywords", [])),
        "cluster_quality_score": round(getattr(controller._data, "cluster_quality_score", -1), 3),
        "next_url": "/akabat/create_db/",
        "back_url": "/akabat/group_keywords/",
    })


def create_db_view(request):
    current_step, total_steps = 7, 9
    csvs_folder = os.path.join(settings.MEDIA_ROOT, "temp_merge_csvs")
    controller = get_controller(request)
    controller._preferences.csv_import_column_mappings_by_file = {
        csv["filename"]: {k: k for k in ["title", "author", "publication_year", "doi", "country", "keywords", "citations"]}
        for csv in request.session.get("csvs", [])
    }
    controller.import_all_csvs(csvs_folder)
    included = request.session.get("included_keywords", {
        "included_starting_by_keywords": [],
        "included_keywords": [],
        "included_contains_keywords": [],
    })
    excluded = request.session.get("excluded_keywords", {
        "excluded_starting_by_keywords_at_csv_import": [],
        "excluded_keywords_at_csv_import": [],
        "excluded_contains_keywords_at_csv_import": [],
        "excluded_keywords_in_plot": [],
    })
    controller._preferences.excluded_keywords_at_csv_import = excluded["excluded_keywords_at_csv_import"]
    controller._preferences.excluded_starting_by_keywords_at_csv_import = excluded["excluded_starting_by_keywords_at_csv_import"]
    controller._preferences.excluded_contains_keywords_at_csv_import = excluded.get("excluded_contains_keywords_at_csv_import", [])
    controller._raw_papers = controller._paper_loader.apply_keyword_exclusions(
        controller._raw_papers,
        included_keywords=included["included_keywords"],
        included_starting_by_keywords=included["included_starting_by_keywords"],
        included_contains_keywords=included["included_contains_keywords"],
        excluded_keywords=excluded["excluded_keywords_at_csv_import"],
        excluded_starting_by_keywords=excluded["excluded_starting_by_keywords_at_csv_import"],
        excluded_contains_keywords=excluded["excluded_contains_keywords_at_csv_import"]
    )
    controller._data.unique_keywords = controller._paper_loader.get_unique_keywords(controller._raw_papers)
    groups, _ = controller.group_keywords_by_semantic_similarity(
        k=None,
        linkage="average",
        affinity="cosine",
        distance_threshold=None
    )
    controller._data.unique_keywords_groups = groups
    author_clusters_path = os.path.join(get_project_folder(request), "author_clusters.json")
    if os.path.exists(author_clusters_path):
        with open(author_clusters_path, "r", encoding="utf-8") as f:
            controller._data.author_clusters = json.load(f)
    else:
        controller._data.author_clusters = {}

    project_folder = get_project_folder(request)
    db_path = os.path.join(project_folder, "db.sqlite3")
    if os.path.exists(db_path):
        os.remove(db_path)
    controller.create_and_populate_database()
    # Show overview of database
    db_size = round(os.path.getsize(db_path) / (1024 * 1024), 2)
    created_at = datetime.fromtimestamp(os.path.getmtime(db_path)).strftime("%Y-%m-%d %H:%M:%S")
    table_info = []

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        for (table_name,) in cursor.fetchall():
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            (row_count,) = cursor.fetchone()
            table_info.append({
                "name": table_name,
                "rows": row_count,
                "columns": len(columns),
            })

    return render(request, "akabat_app/wizard_database.html", {
        "current_step": current_step,
        "total_steps": total_steps,
        "step_title": "Create Database",
        "result_message": "Database created and populated.",
        "db_path": db_path,
        "db_size": db_size,
        "created_at": created_at,
        "table_info": table_info,
        "back_url": "/akabat/cluster_authors/",
        "next_url": "/akabat/plots/",
    })


@require_GET
def get_filter_options(request):
    current_step, total_steps = 8, 9
    csvs_folder = os.path.join(settings.MEDIA_ROOT, "temp_merge_csvs")
    controller = get_controller(request)
    controller._preferences.csv_import_column_mappings_by_file = {
        csv["filename"]: {
            k: k
            for k in ["title", "author", "publication_year", "doi", "country", "keywords", "citations"]
        }
        for csv in request.session.get("csvs", [])
    }
    controller.import_all_csvs(csvs_folder)
    included = request.session.get("included_keywords", {
        "included_starting_by_keywords": [],
        "included_keywords": [],
        "included_contains_keywords": [],
    })
    excluded = request.session.get("excluded_keywords", {
        "excluded_starting_by_keywords_at_csv_import": [],
        "excluded_keywords_at_csv_import": [],
        "excluded_contains_keywords_at_csv_import": [],
        "excluded_keywords_in_plot": [],
    })
    controller._preferences.excluded_keywords_at_csv_import = excluded["excluded_keywords_at_csv_import"]
    controller._preferences.excluded_starting_by_keywords_at_csv_import = excluded["excluded_starting_by_keywords_at_csv_import"]
    controller._preferences.excluded_contains_keywords_at_csv_import = excluded.get("excluded_contains_keywords_at_csv_import", [])
    controller._raw_papers = controller._paper_loader.apply_keyword_exclusions(
        controller._raw_papers,
        included_keywords=included["included_keywords"],
        included_starting_by_keywords=included["included_starting_by_keywords"],
        included_contains_keywords=included["included_contains_keywords"],
        excluded_keywords=excluded["excluded_keywords_at_csv_import"],
        excluded_starting_by_keywords=excluded["excluded_starting_by_keywords_at_csv_import"],
        excluded_contains_keywords=excluded["excluded_contains_keywords_at_csv_import"]
    )
    author_clusters_path = os.path.join(get_project_folder(request), "author_clusters.json")
    with open(author_clusters_path, "r", encoding="utf-8") as f:
        controller._data.author_clusters = json.load(f)
    keyword_groups_path = os.path.join(get_project_folder(request), "keyword_groups.json")
    with open(keyword_groups_path, "r", encoding="utf-8") as f:
        controller._data.unique_keywords_groups = json.load(f)

    raw_df = controller._raw_papers
    years = sorted(set(raw_df["publication_year"].dropna().astype(str)))

    keyword_col = raw_df["keywords"].dropna().astype(str)
    keywords = sorted({kw.strip() for kws in keyword_col for kw in kws.split(";") if kw.strip()})

    author_col = raw_df["author"].dropna()
    authors = sorted({a.strip() for row in author_col for a in row.split(";") if a.strip()})

    keyword_group_names = list(controller._data.unique_keywords_groups.keys())
    author_group_names = list(controller._data.author_clusters.keys())

    return JsonResponse({
        "years": years,
        "keywords": keywords,
        "authors": authors,
        "keyword_groups": keyword_group_names,
        "author_groups": author_group_names,
    })



@require_GET
def load_plot_ajax(request, plot_name):
    current_step, total_steps = 8, 9
    csvs_folder = os.path.join(settings.MEDIA_ROOT, "temp_merge_csvs")
    controller = get_controller(request)
    controller._preferences.csv_import_column_mappings_by_file = {
        csv["filename"]: {
            k: k
            for k in ["title", "author", "publication_year", "doi", "country", "keywords", "citations"]
        }
        for csv in request.session.get("csvs", [])
    }
    controller.import_all_csvs(csvs_folder)
    included = request.session.get("included_keywords", {
        "included_starting_by_keywords": [],
        "included_keywords": [],
        "included_contains_keywords": [],
    })
    excluded = request.session.get("excluded_keywords", {
        "excluded_starting_by_keywords_at_csv_import": [],
        "excluded_keywords_at_csv_import": [],
        "excluded_contains_keywords_at_csv_import": [],
        "excluded_keywords_in_plot": [],
    })
    controller._preferences.excluded_keywords_at_csv_import = excluded["excluded_keywords_at_csv_import"]
    controller._preferences.excluded_starting_by_keywords_at_csv_import = excluded["excluded_starting_by_keywords_at_csv_import"]
    controller._preferences.excluded_contains_keywords_at_csv_import = excluded.get("excluded_contains_keywords_at_csv_import", [])
    controller._raw_papers = controller._paper_loader.apply_keyword_exclusions(
        controller._raw_papers,
        included_keywords=included["included_keywords"],
        included_starting_by_keywords=included["included_starting_by_keywords"],
        included_contains_keywords=included["included_contains_keywords"],
        excluded_keywords=excluded["excluded_keywords_at_csv_import"],
        excluded_starting_by_keywords=excluded["excluded_starting_by_keywords_at_csv_import"],
        excluded_contains_keywords=excluded["excluded_contains_keywords_at_csv_import"]
    )
    author_clusters_path = os.path.join(get_project_folder(request), "author_clusters.json")
    threshold = float(request.GET.get("threshold", 0))
    top_n = int(request.GET.get("top_n", 0)) or None
    min_freq = int(request.GET.get("min_freq", 0))
    filters = {
        "years": request.GET.get("years", "").split(",") if request.GET.get("years") else [],
        "keywords": request.GET.get("keywords", "").split(",") if request.GET.get("keywords") else [],
        "keyword_groups": request.GET.get("keyword_groups", "").split(",") if request.GET.get("keyword_groups") else [],
        "authors": request.GET.get("authors", "").split(",") if request.GET.get("authors") else [],
        "author_groups": request.GET.get("author_groups", "").split(",") if request.GET.get("author_groups") else [],
        "showlegend": request.GET.get("showlegend", "1") == "1"
    }
    with open(author_clusters_path, "r", encoding="utf-8") as f:
        controller._data.author_clusters = json.load(f)
    try:
        fig_dict = controller.generate_single_plot(
            plot_name,
            threshold=threshold,
            top_n=top_n,
            min_freq=min_freq,
            filters=filters  
        )
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse(fig_dict, encoder=PlotlyJSONEncoder)


def plots_view(request):
    current_step, total_steps = 8, 9
    csvs_folder = os.path.join(settings.MEDIA_ROOT, "temp_merge_csvs")
    controller = get_controller(request)
    controller._preferences.csv_import_column_mappings_by_file = {
        csv["filename"]: {
            k: k
            for k in ["title", "author", "publication_year", "doi", "country", "keywords", "citations"]
        }
        for csv in request.session.get("csvs", [])
    }
    controller.import_all_csvs(csvs_folder)
    included = request.session.get("included_keywords", {
        "included_starting_by_keywords": [],
        "included_keywords": [],
        "included_contains_keywords": [],
    })
    excluded = request.session.get("excluded_keywords", {
        "excluded_starting_by_keywords_at_csv_import": [],
        "excluded_keywords_at_csv_import": [],
        "excluded_contains_keywords_at_csv_import": [],
        "excluded_keywords_in_plot": [],
    })
    controller._preferences.excluded_keywords_at_csv_import = excluded["excluded_keywords_at_csv_import"]
    controller._preferences.excluded_starting_by_keywords_at_csv_import = excluded["excluded_starting_by_keywords_at_csv_import"]
    controller._preferences.excluded_contains_keywords_at_csv_import = excluded.get("excluded_contains_keywords_at_csv_import", [])
    controller._raw_papers = controller._paper_loader.apply_keyword_exclusions(
        controller._raw_papers,
        included_keywords=included["included_keywords"],
        included_starting_by_keywords=included["included_starting_by_keywords"],
        included_contains_keywords=included["included_contains_keywords"],
        excluded_keywords=excluded["excluded_keywords_at_csv_import"],
        excluded_starting_by_keywords=excluded["excluded_starting_by_keywords_at_csv_import"],
        excluded_contains_keywords=excluded["excluded_contains_keywords_at_csv_import"]
    )
    author_clusters_path = os.path.join(get_project_folder(request), "author_clusters.json")
    with open(author_clusters_path, "r", encoding="utf-8") as f:
        controller._data.author_clusters = json.load(f)

    graph_definitions = [
        ("polar", "Yearly Distribution of Articles by Category", "time",
        "Angular axis = year. Radial axis = number of papers. Color = category."),
        ("lines", "Year evolution per group category", "time",
        "X = year. Y = number of papers. Color = category. Shows annual trend per group."),
        ("log", "Logaritmic evolution per group category", "time",
        "Same as the yearly line chart but using logarithmic Y-axis to highlight growth patterns."),
        ("map", "Proportion of Publications by Country Over Time", "geo",
        "Map over time. Color = publication share per country and year. Hover shows country and number of publications."),
        ("cumulative_publications", "Cumulative publications by country and year", "geo",
        "Animated map showing cumulative publications per country over time."),
        ("author_cluster", "Authors Semantic Clustering", "author",
        "Nodes = authors. Color = normalized citations. Size = number of articles. Edges = shared cluster. Position = spring layout."),
        ("bubble", "Keywords per Cluster of Authors", "keywords",
        "X = cluster. Y = keyword. Size = frequency. Color = cluster. Interactive Top-N filter available."),
        ("heatmap", "Heatmap of Keywords per Cluster of Authors", "keywords",
        "X = cluster. Y = keyword. Color = frequency. Hover shows keyword and cluster. Filter by minimum frequency."),
        ("impact_map", "Impact Map of Keywords", "geo",
        "Same as publication map, but using citation counts. Color = share of total citations per country and year."),
        ("coauthors", "Coauthorship Network", "author",
        "Nodes = authors. Edges = real coauthorships. Color = normalized degree. Size = number of papers. Labels = top connected nodes."),
        ("cumulative_citations", "Cumulative citations by country and year", "geo",
        "Animated map showing cumulative citation counts per country over time."),
        ("polar_comparison", "Polar Comparison of Categories", "time",
            "Side-by-side polar plots comparing categories across two periods."),

    ]

    graphs_with_info = []
    for name, title, group, explanation in graph_definitions:
        graphs_with_info.append({
            "name": name,
            "div": "",  # no graph loaded initially
            "explanation": explanation,
            "group": group,
            "title": title,
        })

    context = {
        "current_step": current_step,
        "total_steps": total_steps,
        "step_title": "Graphical Analysis",
        "graphs_with_info": graphs_with_info,
        "back_url": "/akabat/create_db/",
        "next_url": "/akabat/save_project/"
    }
    return render(request, "akabat_app/wizard_plots.html", context)


def end_view(request):
    return render(request, "akabat_app/end.html")

def guardar_proyecto(request, paso_actual):
    user_uuid = request.COOKIES.get("user_uuid")
    folder_path = request.session.get("akabat_project_folder", "/tmp/akabat_project")

    if user_uuid:
        name = f"Akabat Project {timezone.now().strftime('%Y-%m-%d %H:%M')}"
        SavedProject.objects.create(
            user_uuid=user_uuid,
            folder_path=folder_path,
            name=name,
            last_step=paso_actual
        )


def save_project_view(request):
    current_step, total_steps = 9, 9
    project_folder = get_project_folder(request)

    prefs = {
        "paths": {
            "root_folder": ".",
            "csv_folder": os.path.join(project_folder, "csvs"),
            "output_files_folder": project_folder,
            "plot_folder": os.path.join(project_folder, "plots"),
        },
        "csv_column_mappings_by_file": {
            csv["filename"]: csv.get("preferences", {})
            for csv in request.session.get("csvs", [])
        },
        "excluded_keywords": request.session["excluded_keywords"],
    }

    with open(os.path.join(project_folder, "preferences.json"), "w", encoding="utf-8") as f:
        json.dump(prefs, f, indent=2, ensure_ascii=False)
    if request.method == "POST":
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(project_folder):
                for fn in files:
                    full = os.path.join(root, fn)
                    rel = os.path.relpath(full, project_folder)
                    zf.write(full, rel)
        buf.seek(0)
        return FileResponse(buf, as_attachment=True, filename="ResultsProject.zip")

    return render(request, "akabat_app/wizard_save.html",  {
        "current_step": current_step,
        "total_steps": total_steps,
        "step_title": "Save Project",
        "result_message": "Your project is ready. Press <strong>Download</strong> to get the ZIP.",
        "download_button": True,
        "end_url":  "/akabat/end/",
    })


def continue_project_view(request):
    if request.method == "POST":
        zip_file = request.FILES.get("project_zip")
        selected_step = request.POST.get("step")

        if not zip_file:
            return HttpResponse("⚠️ No ZIP file uploaded", status=400)
        project_id = uuid.uuid4().hex
        temp_path = os.path.join(settings.MEDIA_ROOT, "projects", project_id)
        os.makedirs(temp_path, exist_ok=True)
        with zipfile.ZipFile(zip_file) as zip_ref:
            zip_ref.extractall(temp_path)
        prefs_path = os.path.join(temp_path, "preferences.json")
        with open(prefs_path, "r", encoding="utf-8") as f:
            prefs = json.load(f)
        request.session["akabat_project_folder"] = temp_path
        request.session["excluded_keywords"] = prefs.get("excluded_keywords", {})
        request.session["column_names"] = prefs.get("csv_import_column_names", {})
        csvs_folder = os.path.join(temp_path, "csvs")
        column_mappings = prefs.get("csv_column_mappings_by_file", {})
        csv_files = []
        for fname in os.listdir(csvs_folder):
            if fname.endswith(".csv"):
                csv_files.append({
                    "id": uuid.uuid4().hex,
                    "filename": fname,
                    "path": os.path.join(csvs_folder, fname),
                    "source": "local",
                    "preferences": column_mappings.get(fname, {
                        "title": "Title",
                        "author": "Author",
                        "publication_year": "Publication Year",
                        "doi": "DOI",
                        "country": "Affiliations",
                        "keywords": "Manual Tags"
                    })
                })
        request.session["csvs"] = csv_files
        step_urls = {
            "import": "/akabat/import/",
            "preferences": "/akabat/preferences/",
            "keywords": "/akabat/process_keywords/",
            "grouping": "/akabat/group_semantic/",
            "clustering": "/akabat/cluster_authors/",
            "plots": "/akabat/plots/",
        }
        return redirect(step_urls.get(selected_step, "/akabat/process_keywords/"))
    return render(request, "akabat_app/continue_project.html")


