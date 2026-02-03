# -*- coding: utf-8 -*-
"""Modern CSS theme for OpenJudge Studio."""

import streamlit as st

# ============================================================================
# Color Palette
# ============================================================================

COLORS = {
    # Primary gradient
    "primary_start": "#6366F1",
    "primary_mid": "#8B5CF6",
    "primary_end": "#D946EF",
    # Background
    "bg_dark": "#0F172A",
    "bg_card": "#1E293B",
    "bg_hover": "#334155",
    # Text
    "text_primary": "#F1F5F9",
    "text_secondary": "#94A3B8",
    "text_muted": "#64748B",
    # Status
    "success": "#22C55E",
    "warning": "#EAB308",
    "error": "#EF4444",
    "info": "#3B82F6",
    # Border
    "border": "#334155",
    "border_active": "#6366F1",
}

# ============================================================================
# Custom CSS
# ============================================================================

CUSTOM_CSS = """
<style>
/* =========================================================================
   Font Import
   ========================================================================= */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

/* =========================================================================
   Global Styles
   ========================================================================= */
.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Reduce top padding in main content area */
.stMainBlockContainer,
.block-container {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* Remove extra top margin from first element */
.stMain .stVerticalBlock > div:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* Hide the header completely to save space */
[data-testid="stHeader"] {
    display: none !important;
}

/* Reduce space between header and main content */
.stAppViewContainer {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* Make the main area start at the very top */
.stMain {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* Reduce top space in the main block container */
[data-testid="stAppViewBlockContainer"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* Remove padding from vertical blocks */
.stVerticalBlock {
    gap: 0.5rem !important;
}

/* Compact the entire app view */
[data-testid="stAppViewContainer"] {
    padding-top: 0 !important;
}

/* Remove top margin from markdown elements */
.stMarkdown {
    margin-top: 0 !important;
}

/* First element in main content should have no top spacing */
/* WARNING: This selector depends on Streamlit's internal DOM structure and may
   break with future Streamlit updates. Unfortunately, Streamlit doesn't provide
   a stable data-testid for this specific element. Monitor after Streamlit upgrades. */
.stMain > div > div > div:first-child {
    padding-top: 0.5rem !important;
}

/* =========================================================================
   Typography
   ========================================================================= */
.main-header {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #D946EF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
    letter-spacing: -0.02em;
}

.sub-header {
    font-size: 1rem;
    color: #94A3B8;
    margin-bottom: 0.75rem;
    font-weight: 400;
}

.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #F1F5F9;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.category-header {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748B;
    margin: 1rem 0 0.5rem 0;
    padding-bottom: 0.25rem;
    border-bottom: 1px solid #334155;
}

/* =========================================================================
   Cards & Containers
   ========================================================================= */
.card {
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}

.card:hover {
    border-color: #6366F1;
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.15);
}

.feature-card {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.05) 100%);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
}

.info-card {
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 8px;
    padding: 0.75rem;
    margin: 0.5rem 0;
}

/* =========================================================================
   Score Display
   ========================================================================= */
.score-container {
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
}

.score-value {
    font-size: 4.5rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.5rem;
}

.score-label {
    font-size: 1rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.score-range-badge {
    display: inline-block;
    font-size: 0.75rem;
    color: #64748B;
    background: #1E293B;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    margin-left: 0.5rem;
}

/* =========================================================================
   Progress Bar
   ========================================================================= */
.progress-container {
    background: #1E293B;
    border-radius: 8px;
    height: 8px;
    overflow: hidden;
    margin: 1rem 0;
}

.progress-bar {
    height: 100%;
    border-radius: 8px;
    transition: width 0.5s ease;
}

/* =========================================================================
   Status Badges
   ========================================================================= */
.status-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.5rem 1rem;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.875rem;
    gap: 0.5rem;
}

.status-pass {
    background: rgba(34, 197, 94, 0.15);
    color: #22C55E;
    border: 1px solid rgba(34, 197, 94, 0.3);
}

.status-fail {
    background: rgba(239, 68, 68, 0.15);
    color: #EF4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.status-warning {
    background: rgba(234, 179, 8, 0.15);
    color: #EAB308;
    border: 1px solid rgba(234, 179, 8, 0.3);
}

/* =========================================================================
   Reason Card
   ========================================================================= */
.reason-card {
    background: #1E293B;
    border-radius: 12px;
    padding: 1.25rem;
    margin-top: 1rem;
    border-left: 4px solid;
}

/* =========================================================================
   Empty State
   ========================================================================= */
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    color: #64748B;
    border: 2px dashed #334155;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.5) 100%);
}

.empty-state-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
    opacity: 0.5;
}

/* =========================================================================
   Guide Steps
   ========================================================================= */
.guide-step {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    padding: 0.75rem 0;
    border-bottom: 1px solid #334155;
}

.guide-step:last-child {
    border-bottom: none;
}

.guide-number {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: linear-gradient(135deg, #6366F1, #8B5CF6);
    border-radius: 50%;
    color: white;
    font-size: 0.875rem;
    font-weight: 600;
    flex-shrink: 0;
}

.guide-text {
    color: #CBD5E1;
    font-size: 0.9rem;
    line-height: 1.5;
}

/* =========================================================================
   Grader Selector
   ========================================================================= */
.grader-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.5rem 0.75rem;
    margin: 0.25rem 0;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s ease;
}

.grader-item:hover {
    background: rgba(99, 102, 241, 0.1);
}

.grader-item.selected {
    background: rgba(99, 102, 241, 0.2);
    border: 1px solid rgba(99, 102, 241, 0.3);
}

.grader-icon {
    font-size: 1.25rem;
}

.grader-name {
    font-weight: 500;
    color: #F1F5F9;
}

.grader-name-zh {
    font-size: 0.8rem;
    color: #94A3B8;
}

/* =========================================================================
   Form Elements
   ========================================================================= */
.stTextArea textarea {
    font-size: 14px !important;
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
    background: #0F172A !important;
    font-family: 'Inter', sans-serif !important;
}

.stTextArea textarea:focus {
    border-color: #475569 !important;
    box-shadow: none !important;
    outline: none !important;
}

.stTextInput input {
    border-radius: 10px !important;
    border: none !important;
    background: #0F172A !important;
    box-shadow: none !important;
}

.stTextInput input:focus {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}

/* Remove wrapper border */
.stTextInput > div {
    border: none !important;
}

.stTextInput [data-baseweb="base-input"] {
    border: none !important;
    background: transparent !important;
}

.stTextInput [data-baseweb="input"] {
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    background: #0F172A !important;
}

.stTextInput [data-baseweb="input"]:focus-within {
    border-color: #334155 !important;
    box-shadow: none !important;
}

.stSelectbox > div > div {
    border-radius: 10px !important;
    border: 1px solid #334155 !important;
    background: #0F172A !important;
}

/* Code editor styling */
.code-editor textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
}

/* =========================================================================
   Button Styling
   ========================================================================= */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    border: none !important;
}

/* =========================================================================
   Tabs - Clean Modern Style
   ========================================================================= */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: transparent;
    border-bottom: none !important;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 0.5rem 1rem;
    background: #1E293B;
    border: 1px solid #334155 !important;
    color: #94A3B8;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.1)) !important;
    border-color: #6366F1 !important;
    color: #F1F5F9 !important;
}

/* Hide ugly tab highlight/underline */
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}

/* Hide tab border line */
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

/* Remove any default underlines in tabs */
.stTabs [role="tablist"] {
    border-bottom: none !important;
}

.stTabs .st-emotion-cache-1kyxreq {
    border-bottom: none !important;
}

/* =========================================================================
   Expander
   ========================================================================= */
.streamlit-expanderHeader {
    background: #1E293B !important;
    border-radius: 10px !important;
    border: 1px solid #334155 !important;
}

.streamlit-expanderHeader:hover {
    border-color: #6366F1 !important;
}

/* =========================================================================
   File Uploader
   ========================================================================= */
.stFileUploader > div {
    border-radius: 12px !important;
    border: 2px dashed #334155 !important;
    background: rgba(15, 23, 42, 0.5) !important;
    padding: 1.5rem !important;
}

.stFileUploader > div:hover {
    border-color: #6366F1 !important;
    background: rgba(99, 102, 241, 0.05) !important;
}

/* =========================================================================
   Image Preview
   ========================================================================= */
.image-preview {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #334155;
    background: #0F172A;
}

.image-preview img {
    max-width: 100%;
    height: auto;
}

/* =========================================================================
   Divider
   ========================================================================= */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #334155, transparent);
    margin: 1.5rem 0;
}

/* =========================================================================
   Footer
   ========================================================================= */
.footer {
    text-align: center;
    color: #475569;
    font-size: 0.8rem;
    padding: 1rem;
    margin-top: 2rem;
}

.footer a {
    color: #6366F1;
    text-decoration: none;
}

.footer a:hover {
    text-decoration: underline;
}

/* =========================================================================
   Animations
   ========================================================================= */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.animate-fade-in {
    animation: fadeIn 0.5s ease-out;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.animate-pulse {
    animation: pulse 2s ease-in-out infinite;
}

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

.loading-shimmer {
    background: linear-gradient(90deg, #1E293B 0%, #334155 50%, #1E293B 100%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
}

/* =========================================================================
   Scrollbar
   ========================================================================= */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #0F172A;
}

::-webkit-scrollbar-thumb {
    background: #334155;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #475569;
}

/* =========================================================================
   Metrics Card
   ========================================================================= */
.metric-card {
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}

.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #F1F5F9;
}

.metric-label {
    font-size: 0.75rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* =========================================================================
   Sidebar Styling
   ========================================================================= */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
}

/* Remove all top padding from sidebar */
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}

/* Hide the original header area completely */
[data-testid="stSidebarHeader"] {
    position: absolute !important;
    top: 0.6rem !important;
    right: 0.5rem !important;
    left: auto !important;
    width: auto !important;
    height: auto !important;
    min-height: 0 !important;
    padding: 0 !important;
    z-index: 100 !important;
}

/* Collapse button styling */
[data-testid="stSidebarCollapseButton"] {
    margin: 0 !important;
}

/* Content area - starts from top */
[data-testid="stSidebarUserContent"],
[data-testid="stSidebarContent"] {
    padding-top: 0.5rem !important;
}

/* Remove any min-height from sidebar inner containers */
[data-testid="stSidebar"] section > div {
    min-height: 0 !important;
}

/* Sidebar header - logo and title inline with space for button */
.sidebar-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.25rem 0;
    padding-right: 2.5rem; /* Space for collapse button */
}

.sidebar-header-text {
    display: flex;
    flex-direction: column;
    justify-content: center;
}

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label {
    color: #94A3B8 !important;
    font-size: 0.85rem !important;
}

/* Hide "No results" autocomplete dropdown for text inputs */
.stTextInput [data-baseweb="popover"] {
    display: none !important;
}

/* =========================================================================
   Clean Up Ugly Default Styles
   ========================================================================= */

/* Remove all default underlines and borders */
hr {
    display: none !important;
}

/* Hide Streamlit's default dividers */
[data-testid="stMarkdownContainer"] hr {
    display: none !important;
}

/* Remove default decorative lines */
.st-emotion-cache-1kyxreq,
.st-emotion-cache-16txtl3,
[class*="st-b"] + [class*="st-c"] {
    border: none !important;
}

/* Clean input focus states - no ugly outlines */
*:focus {
    outline: none !important;
}

/* Remove default Streamlit decoration lines */
.stDeployButton,
[data-testid="stDecoration"] {
    display: none !important;
}

/* Hide hamburger menu decorative line */
[data-testid="stHeader"]::after {
    display: none !important;
}

/* Clean selectbox - remove default borders */
.stSelectbox [data-baseweb="select"] > div {
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
}

.stSelectbox [data-baseweb="select"] > div:hover {
    border-color: #6366F1 !important;
}

/* Clean radio buttons */
.stRadio > div {
    gap: 1rem;
}

.stRadio label {
    padding: 0.5rem 1rem !important;
    border-radius: 8px !important;
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    transition: all 0.2s ease !important;
}

.stRadio label:hover {
    border-color: #6366F1 !important;
}

.stRadio [data-checked="true"] label {
    background: rgba(99, 102, 241, 0.2) !important;
    border-color: #6366F1 !important;
}

/* Clean checkbox */
.stCheckbox label {
    color: #94A3B8 !important;
}

/* Clean number input */
.stNumberInput input {
    border-radius: 10px !important;
    border: 1px solid #334155 !important;
    background: #0F172A !important;
}

.stNumberInput input:focus {
    border-color: #475569 !important;
    box-shadow: none !important;
    outline: none !important;
}

/* Remove all purple focus outlines from inputs */
input:focus,
textarea:focus,
select:focus,
[data-baseweb="input"]:focus-within,
[data-baseweb="textarea"]:focus-within,
[data-baseweb="select"]:focus-within {
    outline: none !important;
    box-shadow: none !important;
}

/* Override Streamlit's default focus styles */
.stTextInput [data-baseweb="input"] {
    border-color: #334155 !important;
}

.stTextInput [data-baseweb="input"]:focus-within {
    border-color: #475569 !important;
    box-shadow: none !important;
}

/* Remove inner input focus ring */
.stTextInput input:focus {
    outline: none !important;
    box-shadow: none !important;
    border: none !important;
}

/* Clean slider */
.stSlider > div > div {
    background: transparent !important;
}

.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    border: none !important;
    width: 16px !important;
    height: 16px !important;
    box-shadow: none !important;
}

/* Slider track */
.stSlider [data-baseweb="slider"] > div:first-child {
    background: #334155 !important;
    height: 6px !important;
    border-radius: 3px !important;
}

/* Slider filled track */
.stSlider [data-baseweb="slider"] > div:first-child > div {
    background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
}

/* Hide tick bar values */
.stSlider [data-testid="stSliderTickBar"] {
    display: none !important;
}

/* Clean slider value display */
.stSlider [data-testid="stTickBar"] {
    display: none !important;
}

/* Slider current value - make it cleaner */
.stSlider > div > div > div:last-child {
    color: #94A3B8 !important;
    font-size: 0.8rem !important;
}

/* Hide min/max labels under slider */
.stSlider > div > div[data-testid="stTickBarMin"],
.stSlider > div > div[data-testid="stTickBarMax"] {
    display: none !important;
}

/* Remove slider track shadow/highlight */
.stSlider [data-baseweb="slider"] > div {
    box-shadow: none !important;
}

/* Remove all slider shadows */
.stSlider * {
    box-shadow: none !important;
}

/* Clean slider track appearance */
.stSlider [data-baseweb="slider"] [role="slider"]::before,
.stSlider [data-baseweb="slider"] [role="slider"]::after {
    display: none !important;
}

/* Remove inner track shadows */
.stSlider .st-fs,
.stSlider .st-ft {
    background: transparent !important;
    box-shadow: none !important;
}

/* Hide the upper track/highlight line completely */
.stSlider [data-baseweb="slider"] > div:nth-child(2),
.stSlider [data-baseweb="slider"] > div:nth-child(3),
.stSlider [data-baseweb="slider"] > div:nth-child(4) {
    display: none !important;
}

/* Hide all extra slider decorations */
.stSlider div[class*="st-f"],
.stSlider div[class*="st-g"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Force hide hover/active track highlight */
.stSlider [data-baseweb="slider"] div:not([role="slider"]):not(:first-child) {
    opacity: 0 !important;
    visibility: hidden !important;
}

/* Clean expander */
.streamlit-expanderHeader {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
}

.streamlit-expanderHeader:hover {
    border-color: #6366F1 !important;
}

.streamlit-expanderContent {
    border: 1px solid #334155 !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
    background: #0F172A !important;
}

/* Clean alert boxes */
.stAlert {
    border-radius: 10px !important;
    border: none !important;
}

/* Clean status widget */
[data-testid="stStatusWidget"] {
    border-radius: 12px !important;
}

/* Remove iframe borders */
iframe {
    border: none !important;
}

/* Clean JSON viewer */
.stJson {
    background: #0F172A !important;
    border-radius: 10px !important;
    border: 1px solid #334155 !important;
}

/* Hide anchor links */
.css-15zrgzn {
    display: none !important;
}

/* Clean up any remaining ugly lines */
[class*="st-emotion-cache"] {
    border-color: #334155 !important;
}

/* Remove bottom border from tab panels */
[role="tabpanel"] {
    border: none !important;
}
</style>
"""


def inject_css() -> None:
    """Inject custom CSS into the Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def get_score_color(score: float, max_score: float = 5.0) -> str:
    """Get color based on score value.

    Args:
        score: The score value
        max_score: Maximum possible score (default 5.0)

    Returns:
        Hex color string
    """
    ratio = score / max_score
    if ratio >= 0.8:
        return COLORS["success"]
    elif ratio >= 0.6:
        return COLORS["warning"]
    else:
        return COLORS["error"]
