"""Shared styling and UI utilities for Meridian Enterprise RAG."""

import streamlit as st

CUSTOM_CSS = """
<style>
/* 1. Hide the Deploy button (all testids across Streamlit versions) */
[data-testid="stAppDeployButton"],
[data-testid="stDeployButton"],
.stDeployButton {
    display: none !important;
}

/* 2. Hide the 3-dot hamburger menu and extra toolbar action container */
#MainMenu,
[data-testid="stMainMenu"],
[data-testid="stMainMenuButton"],
[data-testid="stToolbarActions"] {
    display: none !important;
}

/* 3. Hide the Stop button in the header / status widget */
[data-testid="stStatusWidget"] button,
button[kind="header"] {
    display: none !important;
}

/* 4. Ensure the status widget animated running spinner remains visible */
[data-testid="stStatusWidget"] {
    visibility: visible !important;
    display: inline-flex !important;
}

[data-testid="stStatusWidgetRunningIcon"],
[data-testid="stStatusWidgetRunningManIcon"] {
    visibility: visible !important;
    display: inline-block !important;
}

/* 5. Hide heading hover anchor link icons */
[data-testid="stHeaderActionElements"], .header-anchor {
    display: none !important;
}
</style>
"""


def apply_custom_styles():
    """Apply global enterprise CSS styles to current page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
