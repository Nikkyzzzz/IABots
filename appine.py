import streamlit as st
import pandas as pd
import plotly.express as px
import importlib.util
from io import BytesIO
import base64
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import pyodbc
from email_sender import send_reports_via_email
import re
import requests
import os
import json
from dotenv import load_dotenv
import time
def slugify(text: str) -> str:
    """
    Turn any bot name into a URL‐/ID‐safe lowercase hyphen slug.
    """
    slug = re.sub(r'[^a-z0-9]+', '-', text.lower())
    return slug.strip('-')

# Load environment variables for Zoho
load_dotenv()
ZOHO_CLIENT_ID = os.getenv("ZOHO_CLIENT_ID")
ZOHO_CLIENT_SECRET = os.getenv("ZOHO_CLIENT_SECRET")
ZOHO_REDIRECT_URI = os.getenv("ZOHO_REDIRECT_URI")
ZOHO_ORGANIZATION_ID = os.getenv("ZOHO_ORGANIZATION_ID")
ZOHO_TOKEN_URL = "https://accounts.zoho.in/oauth/v2/token"
ZOHO_AUTH_URL = "https://accounts.zoho.in/oauth/v2/auth"
ZOHO_API_BASE = "https://www.zohoapis.in/books/v3"
TOKENS_FILE = "tokens.json"
ZOHO_SCOPES = "ZohoBooks.fullaccess.all"

# Load logic.py
spec = importlib.util.spec_from_file_location("logic", "logic.py")
logic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logic)

st.set_page_config(layout="wide")
st.sidebar.title("📂 Procurement Bots")
section = st.sidebar.radio("Navigate", ["📊 Summary Dashboard", "📁 Generated Reports", "📧 Send Reports via Email"])

# ================ ZOHO HELPER FUNCTIONS ================
def save_tokens(data):
    data["fetched_at"] = datetime.now().isoformat()
    with open(TOKENS_FILE, "w") as f:
        json.dump(data, f)

def load_tokens():
    if os.path.exists(TOKENS_FILE):
        with open(TOKENS_FILE, "r") as f:
            return json.load(f)
    return {}

def is_token_expired(tokens):
    if not tokens:
        return True
    fetched_at = datetime.fromisoformat(tokens.get("fetched_at", datetime.min.isoformat()))
    return datetime.now() > fetched_at + timedelta(seconds=int(tokens.get("expires_in", 3600)))

def refresh_access_token():
    tokens = load_tokens()
    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        st.error("No refresh token found. Please re-authenticate.")
        return None
    
    data = {
        "grant_type": "refresh_token",
        "client_id": ZOHO_CLIENT_ID,
        "client_secret": ZOHO_CLIENT_SECRET,
        "refresh_token": refresh_token,
    }
    
    res = requests.post(ZOHO_TOKEN_URL, data=data)
    if res.ok:
        new_tokens = res.json()
        new_tokens["refresh_token"] = refresh_token
        save_tokens(new_tokens)
        return new_tokens["access_token"]
    else:
        st.error(f"Failed to refresh token: {res.text}")
        return None

def get_access_token():
    tokens = load_tokens()
    if not tokens:
        return None
    if is_token_expired(tokens):
        return refresh_access_token()
    return tokens.get("access_token")

def exchange_code_for_token(code):
    data = {
        "grant_type": "authorization_code",
        "client_id": ZOHO_CLIENT_ID,
        "client_secret": ZOHO_CLIENT_SECRET,
        "redirect_uri": ZOHO_REDIRECT_URI,
        "code": code,
    }
    
    res = requests.post(ZOHO_TOKEN_URL, data=data)
    if res.ok:
        save_tokens(res.json())
        st.success("✅ Zoho tokens saved successfully!")
        return True
    else:
        st.error(f"❌ Failed to exchange code: {res.text}")
        return False

def fetch_zoho_data(endpoint):
    access_token = get_access_token()
    if not access_token:
        st.error("Access token missing or failed to refresh.")
        return None

    url = f"{ZOHO_API_BASE}/{endpoint}?organization_id={ZOHO_ORGANIZATION_ID}"
    headers = {"Authorization": f"Zoho-oauthtoken {access_token}"}
    res = requests.get(url, headers=headers)

    if res.ok:
        return res.json().get(endpoint, [])
    else:
        st.error(f"Error fetching {endpoint}: {res.status_code} - {res.text}")
        return None

def fetch_data_from_zoho():
    st.info("Fetching data from Zoho Books...")
    
    # Fetch Purchase Orders
    po_data = fetch_zoho_data("purchaseorders")
    if po_data is None:
        return None, None, None
    
    # Fetch Vendors
    vendor_data = fetch_zoho_data("vendors")
    
    # Convert to DataFrames
    po_df = pd.DataFrame(po_data)
    vendor_df = pd.DataFrame(vendor_data) if vendor_data else pd.DataFrame()
    
    # Attempt to fetch purchase requisitions - Zoho might not have this endpoint
    # Instead, we'll use purchase orders as a fallback for both PO and PR data
    pr_df = po_df.copy().rename(columns={
        "po_number": "PR No",
        "created_time": "Creation Date",
        "last_modified_time": "Approval Date",
        "vendor_id": "Vendor Code",
        "vendor_name": "Vendor Name",
        "status": "Status",
    })
    
    # Add dummy PR-specific columns that might be needed by the logic
    pr_df["Created By"] = "User"
    pr_df["Approved By"] = "Approver"
    
    # Rename columns to match expected names in logic
    po_mapping = {
        "po_number": "PO Number",
        "date": "PO Date",
        "vendor_id": "Vendor Code",
        "vendor_name": "Vendor Name",
        "status": "Status",
        "total": "GTotal",
        "created_time": "Creation Date",
        "last_modified_time": "Modification Date",
    }
    
    pr_mapping = {
        "PR No": "PR No",
        "Creation Date": "Creation Date",
        "Approval Date": "Approval Date",
        "Created By": "Created By",
        "Approved By": "Approved By",
        "Vendor Code": "Vendor Code",
        "Vendor Name": "Vendor Name",
    }
    
    po_df.rename(columns=po_mapping, inplace=True)
    pr_df.rename(columns=pr_mapping, inplace=True)
    
    return pr_df, po_df, vendor_df

# ================ DATABASE HELPER FUNCTION ================
@st.cache_data(show_spinner="Connecting to SQL and fetching data...")
def fetch_data_from_sql():
    conn_str = (
        #'DRIVER={SQL Server};'
        'DRIVER={ODBC Driver 18 for SQL Server};'
        'SERVER=my-server123.database.windows.net;'
        'DATABASE=demo;'
        'UID=kush;'
        'PWD=India@123;'
        'Trusted_Connection=no;'
    )
    conn = pyodbc.connect(conn_str)
    pr_df = pd.read_sql('SELECT * FROM [TEST_PR 1 csv]', conn)
    po_df = pd.read_sql('SELECT * FROM [TEST_PO csv]', conn)
    vendor_df = pd.read_sql('SELECT * FROM [TEST_VENDOR_CSV]', conn)
    conn.close()

    pr_df.columns = pr_df.columns.str.replace('_', ' ')
    po_df.columns = po_df.columns.str.replace('_', ' ')
    vendor_df.columns = vendor_df.columns.str.replace('_', ' ')

    return pr_df, po_df, vendor_df

# ================ MAIN APP ================
# --- Data Source Selection ---
st.sidebar.markdown("### Select Data Source")
data_source = st.sidebar.radio("Load Data From", ["Upload Excel Files", "Fetch from Database", "Fetch from Zoho Books"])

# Initialize dataframes to None before any source selection
pr_df, po_df, vendor_df = None, None, None

# Retrieve from session state if already loaded
if 'pr_df' in st.session_state:
    pr_df = st.session_state['pr_df']
if 'po_df' in st.session_state:
    po_df = st.session_state['po_df']
if 'vendor_df' in st.session_state:
    vendor_df = st.session_state['vendor_df']

if data_source == "Upload Excel Files":
    pr_cols = [
        "PR Number", "Creation Date", "Created By", "Item Code", "UoM",
        "Required Quantity", "Status", "Cost Center", "Item Description",
        "Required Date", "Target Date", "Target Quantity",
        "Requester Name", "Approval Date", "Approved By",
    ]
    buf_pr = BytesIO()
    pd.DataFrame(columns=pr_cols).to_excel(buf_pr, index=False, engine="openpyxl")
    buf_pr.seek(0)
    st.sidebar.download_button(
        "📥 Download empty PR template",
        data=buf_pr,
        file_name="pr_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_pr_template"
    )
    po_cols = [
        "purchaseorder_number","Creation Date","Created By","PO Date","Vendor Code","Vendor Name",
        "Item Code","UoM","Quantity","Unit Price","Department","TaxCode","Document Currency",
        "PR Number","GTotal","LineTotal","Total Tax","Open PO Quantity","Open PO Quantity rate",
        "open po value","Series","Description","PO Due Date","Days","Approval Date","Approved By",
    ]
    buf_po = BytesIO()
    pd.DataFrame(columns=po_cols).to_excel(buf_po, index=False, engine="openpyxl")
    buf_po.seek(0)
    st.sidebar.download_button(
        "📥 Download empty PO template",
        data=buf_po,
        file_name="po_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_po_template"
    )
    pr_file = st.sidebar.file_uploader("Upload PR File", type=["xlsx", "xls", "csv"])
    po_file = st.sidebar.file_uploader("Upload PO File", type=["xlsx", "xls", "csv"])

    uploaded_something = False

    if pr_file:
        try:
            if pr_file.name.lower().endswith(".csv"):
                pr_df = pd.read_csv(pr_file)
            else:
                pr_df = pd.read_excel(pr_file)
            st.session_state['pr_df'] = pr_df
            st.success("✅ PR file uploaded successfully.")
            uploaded_something = True
        except Exception as e:
            st.error(f"❌ Failed to read PR file: {e}")
            pr_df = None # Ensure it's None if reading fails

    if po_file:
        try:
            if po_file.name.lower().endswith(".csv"):
                po_df = pd.read_csv(po_file)
            else:
                po_df = pd.read_excel(po_file)
            st.session_state['po_df'] = po_df
            st.success("✅ PO file uploaded successfully.")
            uploaded_something = True
        except Exception as e:
            st.error(f"❌ Failed to read PO file: {e}")
            po_df = None # Ensure it's None if reading fails

    if po_file and po_df is not None:
        # … right after: if po_file and po_df is not None:

# 1) offer a blank template for download
        


        st.sidebar.markdown("### 🗺️ Map your PO file's columns")
        required_po_cols = [
            "purchaseorder_number","Creation Date","Created By","PO Date","Vendor Code","Vendor Name",
            "Item Code","UoM","Quantity","Unit Price","Department","TaxCode","Document Currency",
            "PR Number","GTotal","LineTotal","Total Tax","Open PO Quantity","Open PO Quantity rate",
            "open po value","Series","Description","PO Due Date","Days","Approval Date","Approved By",
        ]

        def normalize(colname):
            return colname.strip().lower().replace(" ", "")

        # 3) Build dicts of normalized → actual names from the uploaded file
        actual_cols = po_df.columns.tolist()
        normalized_to_actual = { normalize(c): c for c in actual_cols }

        # 4) Auto-map any required field whose normalized form matches exactly
        auto_mapping = {}
        for internal in required_po_cols:
            key_norm = normalize(internal)
            if key_norm in normalized_to_actual:
                auto_mapping[internal] = normalized_to_actual[key_norm]

        # 5) Check whether everything was auto-mapped
        if set(auto_mapping) == set(required_po_cols):
            # we have a full auto-mapping!
            po_df = po_df.rename(columns={ user: internal for internal, user in auto_mapping.items() })
            st.session_state['po_df'] = po_df
            st.sidebar.success("✅ All PO columns auto-detected and mapped!")
        else:
            # 6) Build a “remaining” list for manual mapping
            remaining_required = [r for r in required_po_cols if r not in auto_mapping]
            remaining_actual   = [c for c in actual_cols if normalize(c) not in map(normalize, auto_mapping.values())]

            
            mapping = auto_mapping.copy()
            for internal in remaining_required:
                display_label = internal.replace('_', ' ').upper()
                selected = st.sidebar.selectbox(
                    f"Map '{display_label}' →",
                    options=[""] + remaining_actual,
                    key=f"map_po_{internal}"
                )
                mapping[internal] = selected
                if selected:
                    remaining_actual.remove(selected)

            # 7) Halt until user fills in the rest
            if not all(mapping.values()):
                st.sidebar.error("⚠️ Please map **all** required PO fields before running the app.")
                st.stop()

            # 8) Rename & persist
            po_df = po_df.rename(columns={ user: internal for internal, user in mapping.items() })
            st.session_state['po_df'] = po_df
            st.sidebar.success("✅ PO columns mapped successfully!")
    
            # … right after your PO‐mapping code …

    # ─── PR TEMPLATE & MAPPING ──────────────────────────────────────────
    if pr_file and pr_df is not None:
        # 1) blank PR template download
        

        # 2) column‐mapping UI for PR
        st.sidebar.markdown("### 🗺️ Map your PR file's columns")

        required_pr_cols = [
            "PR Number", "Creation Date", "Created By", "Item Code", "UoM",
            "Required Quantity", "Status", "Cost Center", "Item Description",
            "Required Date", "Target Date", "Target Quantity",
            "Requester Name", "Approval Date", "Approved By",
        ]

        def _norm(s): return s.strip().lower().replace(" ", "")
        actual_pr_cols = pr_df.columns.tolist()
        norm_to_actual_pr = { _norm(c): c for c in actual_pr_cols }

        # auto‐map exact matches
        auto_pr = {}
        for tgt in required_pr_cols:
            k = _norm(tgt)
            if k in norm_to_actual_pr:
                auto_pr[tgt] = norm_to_actual_pr[k]

        if set(auto_pr) == set(required_pr_cols):
            # full auto‐map
            pr_df = pr_df.rename(columns={ v:k for k,v in auto_pr.items() })
            st.session_state['pr_df'] = pr_df
            st.sidebar.success("✅ All PR columns auto-detected and mapped!")
        else:
            # manual map the rest
            rem_tgt = [c for c in required_pr_cols if c not in auto_pr]
            rem_act = [c for c in actual_pr_cols if _norm(c) not in { _norm(v) for v in auto_pr.values() }]
            pr_map = auto_pr.copy()

            for tgt in rem_tgt:
                sel = st.sidebar.selectbox(
                    f"Map '{tgt.upper()}' →",
                    options=[""] + rem_act,
                    key=f"map_pr_{tgt}"
                )
                pr_map[tgt] = sel
                if sel: rem_act.remove(sel)

            if not all(pr_map.values()):
                st.sidebar.error("⚠️ Please map **all** PR fields before running.")
                st.stop()

            pr_df = pr_df.rename(columns={ v:k for k,v in pr_map.items() })
            st.session_state['pr_df'] = pr_df
            st.sidebar.success("✅ PR columns mapped successfully!")
    # ─────────────────────────────────────────────────────────────────────

        # Vendor DF is typically derived or less critical for basic ops, can be empty
        if 'vendor_df' not in st.session_state or st.session_state['vendor_df'] is None:
            st.session_state['vendor_df'] = pd.DataFrame()
        vendor_df = st.session_state['vendor_df'] # Update local variable

        if not uploaded_something and (pr_df is None or pr_df.empty) and (po_df is None or po_df.empty):
            st.warning("Please upload at least one file (PR or PO) to proceed.")

elif data_source == "Fetch from Database":
    if st.sidebar.button("Fetch Data from SQL Server"):
        try:
            pr_df, po_df, vendor_df = fetch_data_from_sql()
            st.session_state['pr_df'] = pr_df
            st.session_state['po_df'] = po_df
            st.session_state['vendor_df'] = vendor_df
            st.success("✅ Data fetched successfully from SQL Server.")
        except Exception as e:
            #st.error(f"❌ Failed to fetch data from SQL Server: {e}")
            # Ensure DFs are reset on failure to prevent using old data
            pr_df, po_df, vendor_df = None, None, None
            if 'pr_df' in st.session_state: del st.session_state['pr_df']
            if 'po_df' in st.session_state: del st.session_state['po_df']
            if 'vendor_df' in st.session_state: del st.session_state['vendor_df']
    # Removed st.stop() here, as the check below will handle if data isn't loaded.

elif data_source == "Fetch from Zoho Books":
    tokens = load_tokens()
    
    if not tokens or not tokens.get("access_token"):
        auth_link = (
            f"{ZOHO_AUTH_URL}?scope={ZOHO_SCOPES}&client_id={ZOHO_CLIENT_ID}"
            f"&response_type=code&access_type=offline&redirect_uri={ZOHO_REDIRECT_URI}"
        )
        st.sidebar.markdown(f"[🔐 Click here to authenticate Zoho]({auth_link})")
        code = st.sidebar.text_input("Paste the authorization code here")
        if st.sidebar.button("Exchange Code for Token") and code:
            if exchange_code_for_token(code):
                tokens = load_tokens()
    
    if tokens and tokens.get("access_token"):
        if st.sidebar.button("Fetch Data from Zoho Books"):
            with st.spinner("Fetching data from Zoho Books..."):
                try:
                    pr_df, po_df, vendor_df = fetch_data_from_zoho()
                    if po_df is not None: # Check po_df as it's the base for both
                        st.session_state['pr_df'] = pr_df
                        st.session_state['po_df'] = po_df
                        st.session_state['vendor_df'] = vendor_df
                        st.success("✅ Data fetched successfully from Zoho Books.")
                    else:
                        st.error("❌ Failed to fetch data from Zoho Books.")
                        # Reset DFs on failure
                        pr_df, po_df, vendor_df = None, None, None
                        if 'pr_df' in st.session_state: del st.session_state['pr_df']
                        if 'po_df' in st.session_state: del st.session_state['po_df']
                        if 'vendor_df' in st.session_state: del st.session_state['vendor_df']
                except Exception as e:
                    st.error(f"❌ Error fetching from Zoho: {str(e)}")
                    # Reset DFs on failure
                    pr_df, po_df, vendor_df = None, None, None
                    if 'pr_df' in st.session_state: del st.session_state['pr_df']
                    if 'po_df' in st.session_state: del st.session_state['po_df']
                    if 'vendor_df' in st.session_state: del st.session_state['vendor_df']

# --- Check if any data is loaded before proceeding with bots/reports ---
# This single check replaces the multiple st.stop() calls after data loading sections
data_loaded_successfully = (pr_df is not None and not pr_df.empty) or \
                           (po_df is not None and not po_df.empty)

if not data_loaded_successfully:
    st.info("Please upload files or fetch data from a source to view reports and dashboards.")
    st.stop() # Stop the execution here if no data is available

# --- Ensure DataFrames are present for bot execution (as empty DFs if needed) ---
# This block runs only if data_loaded_successfully is True, meaning at least one DF has data.
# However, for bots that depend on *both* PR and PO, if only one is available, the other
# should be an empty DataFrame to avoid errors in logic.py
if pr_df is None:
    pr_df = pd.DataFrame()
    st.session_state['pr_df'] = pr_df
if po_df is None:
    po_df = pd.DataFrame()
    st.session_state['po_df'] = po_df
if vendor_df is None:
    vendor_df = pd.DataFrame()
    st.session_state['vendor_df'] = vendor_df

# --- Bot Registry ---
bot_registry = {
    "Purchase Requitition (PR)": {
        "Missing Critical Fields": lambda: logic.get_prs_with_missing_supporting_fields_pr_df(pr_df),
        "Created & Approved by Same User": lambda: logic.filter_same_creator_approver_pr(pr_df, "Created By", "Approved By"),
        "Created Outside Business Hours": lambda: logic.flag_prs_outside_business_hours(pr_df, "Creation Date"),
        "PR without Approver": lambda: logic.flag_missing_approver_ids_pr(pr_df),
        "Backdated/Outside Business Hours Approval": lambda: logic.detect_backdated_pr_approvals(pr_df),
        "UoM Inconsistency": lambda: logic.detect_uom_inconsistency_PR(pr_df),
    },
    "Purchase Order (PO)": {
        "Missing Critical Fields": lambda: logic.get_pos_with_missing_supporting_fields_po_df(po_df),
        "Backdated PO Creation": lambda: logic.filter_backdated_pos(po_df, "PO Date", "Creation Date"),
        "Potential PO Splitting": lambda: logic.detect_po_splitting(po_df),
        "PO Price Spikes": lambda: logic.flag_po_price_spikes(po_df),
        "Created & Approved by Same User": lambda: logic.filter_same_creator_approver_po(po_df, "Created By", "Approved By"),
        "Created Outside Business Hours": lambda: logic.flag_pos_outside_business_hours(po_df, "Creation Date"),
        "PO without Approver": lambda: logic.flag_missing_approver_ids_po(po_df),
        "Backdated/Otuside Business Hours Approvals": lambda: logic.detect_backdated_po_approvals(po_df),
        "UoM Inconsistency": lambda: logic.detect_uom_inconsistency(po_df),
    },
    "PO vs PR": {
        "Delay in PR to PO": lambda: logic.find_delayed_pr_to_po(pr_df, po_df),
        "PR without PO": lambda: logic.find_open_pr_ageing(pr_df, po_df),
        "PR vs PO Item Code Mismatch": lambda: logic.detect_item_code_mismatch(pr_df, po_df),
        "PO Created before PR Approval": lambda: logic.detect_po_created_before_pr_approval(pr_df, po_df),
        "PO & PR Created by Same User": lambda: logic.detect_same_user_pr_po(pr_df, po_df),
        "PO Without PR": lambda: logic.detect_po_without_pr(po_df)
    }
}

# --- Rest of the App ---
# Sidebar Category Selection
selected_category = st.sidebar.selectbox("Choose Report Category", list(bot_registry.keys()))
selected_bots = list(bot_registry[selected_category].keys())

# Sidebar Anchors to Scroll
st.sidebar.markdown("---")
st.sidebar.markdown("### Jump to Bot Result:")
for bot in selected_bots:
    anchor = slugify(bot)      # ← use slugify here
    st.sidebar.markdown(
        f'<a href="#{anchor}"><button style="width:100%">{bot}</button></a>',
        unsafe_allow_html=True
    )

# 📊 Summary Dashboard
if section == "📊 Summary Dashboard":
    st.title("📊 Summary Overview")
    
    # This is the main change: only show summary if data is loaded
    if data_loaded_successfully:
        summary_rows = []

        for bot_name, bot_func in bot_registry[selected_category].items():
            try:
                df = bot_func()
                issue_count = len(df) if isinstance(df, pd.DataFrame) else 0
                summary_rows.append({"Bot": bot_name, "Issues Found": issue_count})
            except Exception as e:
                summary_rows.append({"Bot": bot_name, "Issues Found": "❌ Error"})

        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df)

        if not summary_df.empty and summary_df["Issues Found"].apply(lambda x: isinstance(x, int)).any():
            fig = px.bar(summary_df[summary_df["Issues Found"].apply(lambda x: isinstance(x, int))],
                         x="Bot", y="Issues Found", title="Issues per Bot")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please load data from the sidebar to view the summary dashboard.")


# 📁 Generated Reports
if section == "📁 Generated Reports":
    st.title("📁 Bot Reports")

    # This is also updated: only show reports if data is loaded
    if data_loaded_successfully:
        for bot_name, bot_func in bot_registry[selected_category].items():
            anchor = slugify(bot_name)
            st.markdown(f'<h3 id="{anchor}">{bot_name}</h3>', unsafe_allow_html=True)

            try:
                df = bot_func()
                if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                    st.dataframe(df)
                    for col in df.select_dtypes(include=['datetimetz']):
                        df[col] = df[col].dt.tz_localize(None)
                    output = BytesIO()
                    df.to_excel(output, index=False, engine='openpyxl')
                    output.seek(0)
                    b64 = base64.b64encode(output.read()).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{bot_name}.xlsx">📥 Download {bot_name}</a>'
                    st.markdown(href, unsafe_allow_html=True)

                    if st.button(f"Push Observation - {bot_name}", key=f"push_{bot_name}"):
                        with st.spinner("📤 Pushing observation..."):
                            try:
                                # Convert df to in-memory Excel file
                                output = BytesIO()
                                df.to_excel(output, index=False, engine='openpyxl')
                                output.seek(0)

                                # Files payload
                                files = [
                                    ('attachments[]', (
                                        f"{bot_name}.xlsx",
                                        output,
                                        'application/octet-stream'
                                    ))
                                ]

                                # Payload values from df/report context
                                payload = {
                                    'audit_type': 'general',
                                    'due_date': datetime.today().strftime('%Y-%m-%d'),
                                    'created_by': 'Super Admin',
                                    'organization_name': 'Capitall Consultancy Services Ltd',
                                    'company': 'Capitall Consultancy Services Limited',
                                    'locations': 'Delhi',
                                    'financial_year': '2024-25',
                                    'quarter': 'Q3',
                                    'observation_heading': bot_name,
                                    'observation_description': bot_name,
                                    'reviewer_responsible': 'Atishay Jain'
                                }

                                headers = {
                                    'x-api-key': '47fca1d2b8e14eab9d6c463a8fbe5c23',
                                    'x-api-secret-key': 'WlYHPxu0Ut826ejmyGi6HwKQnUtKo8LxR0LU3wwj'
                                }

                                # POST request
                                response = requests.post(
                                    "https://kkc.grc.capitall.io/api/observation/create",
                                    files=files,
                                    data=payload,
                                    headers=headers
                                )

                                # Parse and show response
                                try:
                                    json_response = response.json()
                                except Exception:
                                    json_response = {"status": str(response.status_code), "msg": response.text}

                                st.markdown("### ✅ API Response")
                                st.json(json_response)

                            except Exception as e:
                                st.error(f"❌ Exception occurred: {str(e)}")

                else:
                    st.info("✅ No issues found.")
            except Exception as e:
                st.error(f"❌ Error in '{bot_name}': {e}")
    else:
        st.info("Please load data from the sidebar to view generated reports.")


# 📧 Send Reports via Email
if section == "📧 Send Reports via Email":
    st.title("📧 Send Reports via Email")

    # This is also updated: only show email form if data is loaded
    if data_loaded_successfully:
        with st.form("email_form"):
            recipient = st.text_input("Recipient Email", placeholder="example@email.com")
            category_option = st.radio("Select What to Send", ["All Categories", "Selected Category"])
            selected_cat = st.selectbox("Choose Category (if selected)", list(bot_registry.keys())) if category_option == "Selected Category" else None
            submitted = st.form_submit_button("Send Email")

            if submitted:
                with st.spinner("Generating reports and sending email..."):
                    report_dict = {}
                    error_log = {}
                    try:
                        target_registry = bot_registry if category_option == "All Categories" else {selected_cat: bot_registry[selected_cat]}
                        for category, bots in target_registry.items():
                            for bot_name, bot_func in bots.items():
                                safe_name = re.sub(r'[\\/*?:\\[\\]/]', '_', f"{category} - {bot_name}" if category_option == "All Categories" else bot_name)[:31]
                                try:
                                    df = bot_func()
                                    if isinstance(df, pd.DataFrame) and not df.empty:
                                        report_dict[safe_name] = df
                                except Exception as e:
                                    error_log[safe_name] = pd.DataFrame({"Error": [str(e)]})
                        report_dict.update(error_log)
                        if not report_dict:
                            st.warning("No data to send. All bots returned empty or errored.")
                        else:
                            success, message = send_reports_via_email(recipient, report_dict)
                            if success:
                                st.success(message)
                            else:
                                st.error(message)
                    except Exception as e:
                        st.error(f"❌ Failed while preparing reports: {e}")
    else:
        st.info("Please load data from the sidebar to send reports via email.")
