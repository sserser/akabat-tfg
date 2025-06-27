from django import forms

class MultiFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True

class CSVUploadForm(forms.Form):
    csv_files = forms.FileField(
    widget=MultiFileInput(attrs={'multiple': True}),
    label="Select Files (CSV)",
    help_text="Select one or more CSV."
     )


ORDERING_CHOICES = [
    ("relevance", "Relevance"),
    ("date", "Most Recent"),
]

KEYWORD_LOGIC_CHOICES = [
    ("AND", "AND"),
    ("OR", "OR"),
]

LANGUAGE_CHOICES = [
    ("", "Any"),
    ("english", "English"),
    ("spanish", "Spanish"),
    ("german", "German"),
    ("french", "French"),
    ("italian", "Italian"),
    ("portuguese", "Portuguese"),
    ("russian", "Russian"),
    ("chinese", "Chinese"),
    ("japanese", "Japanese"),
    ("arabic", "Arabic"),
    ("turkish", "Turkish"),
    ("dutch", "Dutch"),
    ("polish", "Polish"),
    ("korean", "Korean"),
    ("swedish", "Swedish"),
    ("czech", "Czech"),
    ("norwegian", "Norwegian"),
    ("danish", "Danish"),
    ("hungarian", "Hungarian"),
    ("romanian", "Romanian"),
    ("ukrainian", "Ukrainian"),
]

class ScopusSearchForm(forms.Form):
    keywords = forms.CharField(required = False, widget=forms.HiddenInput())

    keyword_logic = forms.ChoiceField(
        choices=KEYWORD_LOGIC_CHOICES,
        required=False,
        label="Combine keywords with",
        initial="OR",
        widget=forms.Select(attrs={"class": "form-select"})
    )

    year_from = forms.IntegerField(
        required=False, label="From Year",
        widget=forms.NumberInput(attrs={"class": "form-control"})
    )
    year_to = forms.IntegerField(
        required=False, label="To Year",
        widget=forms.NumberInput(attrs={"class": "form-control"})
    )
    author = forms.CharField(
        required=False, label="Author",
        widget=forms.TextInput(attrs={"class": "form-control"})
    )
    language = forms.ChoiceField(
        choices=LANGUAGE_CHOICES, required=False, label="Language",
        widget=forms.Select(attrs={"class": "form-select"})
    )
    order_by = forms.ChoiceField(
        choices=ORDERING_CHOICES, required=False, label="Order by",
        widget=forms.Select(attrs={"class": "form-select"})
    )
    max_results = forms.IntegerField(
        initial=25, min_value=1, max_value=2000, label="Max Results",
        widget=forms.NumberInput(attrs={"class": "form-control"})
    )

    advanced_query = forms.CharField(
        required=False,
        label="Advanced Query (optional)",
        widget=forms.Textarea(attrs={
            "class": "form-control",
            "rows": 4,
            "placeholder": "Paste a valid Scopus query here."
        })
    )
