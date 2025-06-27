# akabat_app/urls.py
from django.urls import path
from .views import (
    welcome_view,
    home_view, 
    import_csv_view, 
    process_keywords_view, 
    group_semantic_view, 
    create_db_view,
    preferences_view,
    plots_view,
    save_project_view,
    end_view,
    preview_raw_papers_view,
    continue_project_view,
    cluster_authors_view,
    load_plot_ajax,
    resume_project_view, 
    project_list_view,
    review_import_view,
    get_filter_options
)

urlpatterns = [
    path("", home_view, name="home"),
    path("welcome/", welcome_view, name="welcome"),
    path("import/", import_csv_view, name="upload_csv"),
    path("preferences/", preferences_view, name="preferences"),
    path("process_keywords/", process_keywords_view, name="process_keywords"),
    path("group_semantic/", group_semantic_view, name="group_semantic"),
    path("cluster_authors/", cluster_authors_view, name="cluster_authors"), 
    path("create_db/", create_db_view, name="create_db"),
     path("load_plot/<str:plot_name>/",load_plot_ajax, name="load_plot_ajax"),
    path("plots/", plots_view, name="plots"),
    path("save_project/", save_project_view, name="save_project"),
    path("resume/<int:project_id>/", resume_project_view, name="resume_project"),
    path("review/", review_import_view, name="review_import"),
    path("end/", end_view, name="end"),
    path("preview_raw/", preview_raw_papers_view, name="preview_raw"),
    path("continue/", continue_project_view, name="continue_project"),   
    path("projects/", project_list_view, name="project_list"),
    path("get_filter_options/", get_filter_options, name="get_filter_options"),

]
