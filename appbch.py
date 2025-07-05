import streamlit as st
import os
import json
import requests
from dotenv import load_dotenv
import pandas as pd
from io import BytesIO
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()
CLIENT_ID = os.getenv("ZOHO_CLIENT_ID")
CLIENT_SECRET = os.getenv("ZOHO_CLIENT_SECRET")
REDIRECT_URI = os.getenv("ZOHO_REDIRECT_URI")
ORG_ID = os.getenv("ZOHO_ORGANIZATION_ID")

TOKEN_URL = "https://accounts.zoho.in/oauth/v2/token"
AUTH_URL = "https://accounts.zoho.in/oauth/v2/auth"
API_BASE = "https://www.zohoapis.in/books/v3"
TOKENS_FILE = "tokens.json"
SCOPES = "ZohoBooks.fullaccess.all"

# ----------------- Token Management -----------------
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
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "refresh_token": refresh_token,
    }
    res = requests.post(TOKEN_URL, data=data)
    if res.ok:
        new_tokens = res.json()
        new_tokens["refresh_token"] = refresh_token  # Zoho doesn't always return it again
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
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
        "code": code,
    }
    res = requests.post(TOKEN_URL, data=data)
    if res.ok:
        save_tokens(res.json())
        st.success("Tokens saved successfully!")
    else:
        st.error(f"Failed to exchange code: {res.text}")

# ----------------- API Fetch Logic -----------------
def fetch_purchase_orders():
    access_token = get_access_token()
    if not access_token:
        st.error("Access token missing or failed to refresh.")
        return None

    url = f"{API_BASE}/purchaseorders?organization_id={ORG_ID}"
    headers = {"Authorization": f"Zoho-oauthtoken {access_token}"}
    res = requests.get(url, headers=headers)

    st.subheader("🔍 API Debug Info")
    st.code(f"Status Code: {res.status_code}")
    st.code(res.text)

    if res.ok:
        return res.json().get("purchaseorders", [])
    else:
        st.error("Error fetching purchase orders")
        return None

# ----------------- Excel Export -----------------
def convert_to_excel(purchase_orders):
    if not purchase_orders:
        return None

    df = pd.DataFrame(purchase_orders)
    if df.shape[1] >= 3:
        df.columns.values[1] = "Vendor Code"
        df.columns.values[2] = "Vendor Name"

    output = BytesIO()
    df.to_excel(output, index=False, engine='openpyxl')
    output.seek(0)
    return output

# ----------------- Streamlit UI -----------------
st.title("📦 Zoho Purchase Order Extractor")

tokens = load_tokens()

if "access_token" not in tokens:
    auth_link = (
        f"{AUTH_URL}?scope={SCOPES}&client_id={CLIENT_ID}"
        f"&response_type=code&access_type=offline&redirect_uri={REDIRECT_URI}"
    )
    st.markdown(f"[🔐 Click here to authenticate Zoho]({auth_link})")
    code = st.text_input("Paste the authorization code here")
    if st.button("Exchange Code for Token"):
        if code:
            exchange_code_for_token(code)
        else:
            st.warning("Please enter the code.")
else:
    st.success("✅ Authenticated with Zoho!")
    if st.button("📥 Fetch & Export Purchase Orders"):
        data = fetch_purchase_orders()
        excel_file = convert_to_excel(data)
        if excel_file:
            st.download_button(
                label="⬇️ Download Excel",
                data=excel_file,
                file_name="purchase_orders.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.success("Excel file ready for download!")
        else:
            st.warning("No data found.")



# import streamlit as st
# import os
# import json
# import requests
# from dotenv import load_dotenv
# import pandas as pd
# from io import BytesIO

# # Load environment variables
# load_dotenv()
# CLIENT_ID = os.getenv("ZOHO_CLIENT_ID")
# CLIENT_SECRET = os.getenv("ZOHO_CLIENT_SECRET")
# REDIRECT_URI = os.getenv("ZOHO_REDIRECT_URI")
# ORG_ID = os.getenv("ZOHO_ORGANIZATION_ID")
# TOKEN_URL = "https://accounts.zoho.in/oauth/v2/token"
# AUTH_URL = "https://accounts.zoho.in/oauth/v2/auth"
# API_BASE = "https://www.zohoapis.in/books/v3"
# TOKENS_FILE = "tokens.json"
# SCOPES = "ZohoBooks.fullaccess.all"

# def save_tokens(data):
#     with open(TOKENS_FILE, "w") as f:
#         json.dump(data, f)

# def load_tokens():
#     if os.path.exists(TOKENS_FILE):
#         with open(TOKENS_FILE, "r") as f:
#             return json.load(f)
#     return {}

# def get_access_token():
#     tokens = load_tokens()
#     if "access_token" in tokens:
#         return tokens["access_token"]
#     return None

# def exchange_code_for_token(code):
#     data = {
#         "grant_type": "authorization_code",
#         "client_id": CLIENT_ID,
#         "client_secret": CLIENT_SECRET,
#         "redirect_uri": REDIRECT_URI,
#         "code": code,
#     }
#     res = requests.post(TOKEN_URL, data=data)
#     if res.ok:
#         save_tokens(res.json())
#         st.success("Tokens saved successfully!")
#     else:
#         st.error("Failed to exchange code.")

# def fetch_purchase_orders():
#     access_token = get_access_token()
#     if not access_token:
#         st.error("Access token missing.")
#         return None

#     url = f"{API_BASE}/purchaseorders?organization_id={ORG_ID}"
#     headers = {"Authorization": f"Zoho-oauthtoken {access_token}"}
#     res = requests.get(url, headers=headers)
#     if res.ok:
#         return res.json().get("purchaseorders", [])
#     else:
#         st.error("Error fetching purchase orders")
#         return None

# def convert_to_excel(purchase_orders):
#     if not purchase_orders:
#         return None

#     df = pd.DataFrame(purchase_orders)

#     # Rename 2nd and 3rd columns if they exist
#     if df.shape[1] >= 3:
#         df.columns.values[1] = "Vendor Code"
#         df.columns.values[2] = "Vendor Name"

#     output = BytesIO()
#     df.to_excel(output, index=False, engine='openpyxl')
#     output.seek(0)
#     return output

# # --- Streamlit UI ---
# st.title("📦 Zoho Purchase Order Extractor")

# tokens = load_tokens()

# if "access_token" not in tokens:
#     auth_link = (
#         f"{AUTH_URL}?scope={SCOPES}&client_id={CLIENT_ID}"
#         f"&response_type=code&access_type=offline&redirect_uri={REDIRECT_URI}"
#     )
#     st.markdown(f"[🔐 Click here to authenticate Zoho]({auth_link})")
#     code = st.text_input("Paste the authorization code here")
#     if st.button("Exchange Code for Token"):
#         if code:
#             exchange_code_for_token(code)
#         else:
#             st.warning("Please enter the code.")
# else:
#     st.success("✅ Authenticated with Zoho!")
#     if st.button("📥 Fetch & Export Purchase Orders"):
#         data = fetch_purchase_orders()
#         excel_file = convert_to_excel(data)
#         if excel_file:
#             st.download_button(
#                 label="⬇️ Download Excel",
#                 data=excel_file,
#                 file_name="purchase_orders.xlsx",
#                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#             )
#             st.success("Excel file ready for download!")
#         else:
#             st.error("No purchase orders found or an error occurred.")
