# Auto-generated logic.py with valid bots only
from datetime import datetime
# Category: Purchase Order (PO) vs Purchase Requitition (PR)
# Bot: Delay in PR to PO
def find_delayed_pr_to_po(pr_df, po_df):
    

    # Rename columns to avoid conflicts
    pr_df = pr_df.rename(columns={'Creation Date': 'PR Creation Date'})
    po_df = po_df.rename(columns={'Creation Date': 'PO Creation Date'})
    
    # Convert date columns to datetime
    pr_df['Approval Date'] = pd.to_datetime(pr_df['Approval Date'])
    po_df['PO Creation Date'] = pd.to_datetime(po_df['PO Creation Date'])
    
    # Get the earliest PO creation date for each PR
    earliest_po_dates = po_df.groupby('PR Number')['PO Creation Date'].min().reset_index()
    
    # Merge with PR data
    merged_df = pd.merge(pr_df, earliest_po_dates, on='PR Number', how='inner')
    
    # Calculate delay
    merged_df['Delay (days)'] = (merged_df['PO Creation Date'] - merged_df['Approval Date']).dt.days
    
    # Filter where delay > 30
    delayed_df = merged_df[merged_df['Delay (days)'] > 30].copy()

    # Add ageing bucket
    def categorize_ageing(days):
        if days > 180:
            return '>180d'
        elif days > 90:
            return '>90d'
        elif days > 60:
            return '>60d'
        else:
            return '>30d'
    
    delayed_df['Ageing'] = delayed_df['Delay (days)'].apply(categorize_ageing)
    
    return delayed_df



# Category: Purchase Requitition (PR)
# Bot: Open PR Ageing
def find_open_pr_ageing(pr_df, po_df):



    # Ensure date column is datetime
    pr_df['Approval Date'] = pd.to_datetime(pr_df['Approval Date'], errors='coerce')

    # Filter out PRs with no approval date
    pr_df = pr_df[pr_df['Approval Date'].notna()]

    # Get only PRs that are not present in PO dump
    open_pr_df = pr_df[~pr_df['PR Number'].isin(po_df['PR Number'])].copy()

    # Calculate delay from approval date to today
    open_pr_df['Delay (days)'] = (datetime.now() - open_pr_df['Approval Date']).dt.days

    # Filter PRs where delay is greater than 30 days
    delayed_open_pr_df = open_pr_df[open_pr_df['Delay (days)'] > 30].copy()

    # Add ageing bucket
    def categorize_ageing(days):
        if days > 180:
            return '>180d'
        elif days > 90:
            return '>90d'
        elif days > 60:
            return '>60d'
        else:
            return '>30d'

    delayed_open_pr_df['Ageing'] = delayed_open_pr_df['Delay (days)'].apply(categorize_ageing)

    return delayed_open_pr_df


# Category: Purchase Requitition (PR)
# Bot: UoM Inconsistency in PR
def detect_uom_inconsistency_PR(pr_df, item_col='Item Code', uom_col='UoM'):

    """
    Detects materials requested with inconsistent units of measure (UOMs).
    Returns PR rows where a material was requested with more than one distinct UOM (case-insensitive).

    Parameters:
        pr_df (pd.DataFrame): DataFrame containing PR line item data.
        item_col (str): Column name for material/item ID.
        uom_col (str): Column name for unit of measure.

    Returns:
        pd.DataFrame: Filtered DataFrame with inconsistent UOMs, UOM list, and UOM count.
    """

    # Check for required columns
    if item_col not in pr_df.columns or uom_col not in pr_df.columns:
        raise ValueError(f"Missing expected columns: '{item_col}' or '{uom_col}'")

    # Drop rows with missing Material ID or UOM
    df = pr_df.dropna(subset=[item_col, uom_col]).copy()

    # Normalize UOM for case-insensitive comparison
    df['UOM_normalized'] = df[uom_col].str.strip().str.lower()

    # Group by Material ID and aggregate distinct UOMs
    uom_group = df.groupby(item_col)['UOM_normalized'].agg([
        ('UOM Count', 'nunique'),
        ('UOM List', lambda x: sorted(x.unique()))
    ]).reset_index()

    # Filter materials with more than one UOM
    inconsistent_materials = uom_group[uom_group['UOM Count'] > 1]

    # Merge back with original data for full detail
    result_df = pd.merge(
        df,
        inconsistent_materials[[item_col, 'UOM Count', 'UOM List']],
        on=item_col,
        how='inner'
    ).drop(columns=['UOM_normalized']).sort_values(by=item_col)

    return result_df


# Category: Purchase Order (PO)
# Bot: UoM Inconsistency in PO
def detect_uom_inconsistency(po_df, item_col='Item Code', uom_col='UoM'):

    """
    Detects materials requested with inconsistent units of measure (UOMs).
    Returns PO rows where a material was requested with more than one distinct UOM (case-insensitive).

    Parameters:
        po_df (pd.DataFrame): DataFrame containing PR line item data.
        item_col (str): Column name for material/item ID.
        uom_col (str): Column name for unit of measure.

    Returns:
        pd.DataFrame: Filtered DataFrame with inconsistent UOMs, UOM list, and UOM count.
    """

    # Check for required columns
    if item_col not in po_df.columns or uom_col not in po_df.columns:
        raise ValueError(f"Missing expected columns: '{item_col}' or '{uom_col}'")

    # Drop rows with missing Material ID or UOM
    df = po_df.dropna(subset=[item_col, uom_col]).copy()

    # Normalize UOM for case-insensitive comparison
    df['UOM_normalized'] = df[uom_col].str.strip().str.lower()

    # Group by Material ID and aggregate distinct UOMs
    uom_group = df.groupby(item_col)['UOM_normalized'].agg([
        ('UOM Count', 'nunique'),
        ('UOM List', lambda x: sorted(x.unique()))
    ]).reset_index()

    # Filter materials with more than one UOM
    inconsistent_materials = uom_group[uom_group['UOM Count'] > 1]

    # Merge back with original data for full detail
    result_df = pd.merge(
        df,
        inconsistent_materials[[item_col, 'UOM Count', 'UOM List']],
        on=item_col,
        how='inner'
    ).drop(columns=['UOM_normalized']).sort_values(by=item_col)

    return result_df

# Category: Purchase Order (PO) vs Purchase Requitition (PR)
# Bot: PO without PR
def detect_po_without_pr(po_df, pr_reference_col='PR Number', po_type_col=None, po_types_requiring_pr=None):
    """
    Identifies POs created without a corresponding PR (i.e., null/blank PR Reference).
    
    Parameters:
        po_df (pd.DataFrame): DataFrame containing PO data.
        pr_reference_col (str): Column name in PO data for PR reference.
        po_type_col (str or None): Optional. Column indicating PO type/category.
        po_types_requiring_pr (list or None): List of PO types that require PRs. If None, all are considered.

    Returns:
        pd.DataFrame: Filtered POs that were created without a PR.
    """

    if pr_reference_col not in po_df.columns:
        raise ValueError(f"Missing expected column: '{pr_reference_col}' in PO data.")

    # If specific PO types are provided, filter first
    if po_type_col and po_types_requiring_pr:
        po_df = po_df[po_df[po_type_col].isin(po_types_requiring_pr)]

    # Identify rows where PR Reference is blank/null
    po_without_pr = po_df[
        po_df[pr_reference_col].isna() | 
        (po_df[pr_reference_col].astype(str).str.strip() == '')
    ].copy()

    return po_without_pr


# Category: Purchase Order (PO) vs Purchase Requitition (PR)
# Bot: PO Created before PR Approval
def detect_po_created_before_pr_approval(pr_df, po_df):


    # Rename to avoid conflicts
    pr_df = pr_df.rename(columns={'Creation Date': 'PR Creation Date'})
    po_df = po_df.rename(columns={'Creation Date': 'PO Creation Date'})

    # Convert date columns
    pr_df['Approval Date'] = pd.to_datetime(pr_df['Approval Date'], errors='coerce')
    po_df['PO Creation Date'] = pd.to_datetime(po_df['PO Creation Date'], errors='coerce')

    # Get earliest PO creation date per PR
    earliest_po_dates = po_df.groupby('PR Number')['PO Creation Date'].min().reset_index()

    # Merge with PR data
    merged_df = pd.merge(pr_df, earliest_po_dates, on='PR Number', how='inner')

    # Filter: PO created before PR approval
    result_df = merged_df[merged_df['PO Creation Date'] < merged_df['Approval Date']].copy()

    # Add ageing column (how many days before approval)
    result_df['Days Before Approval'] = (
        result_df['Approval Date'] - result_df['PO Creation Date']
    ).dt.days

    return result_df


# Category: Purchase Order (PO) vs Purchase Requitition (PR)
# Bot: PO & PR Created by Same User
def detect_same_user_pr_po(pr_df, po_df):


    # Rename overlapping columns
    pr_df = pr_df.rename(columns={
        'PR Number': 'PR Number',
        'Created By': 'PR Created By',
        'Creation Date': 'PR Creation Date'
    })
    po_df = po_df.rename(columns={
        'PR Number': 'PR Number',
        'Created By': 'PO Created By',
        'Creation Date': 'PO Creation Date'
    })

    # Drop rows where Created By is null
    pr_df = pr_df.dropna(subset=['PR Created By'])
    po_df = po_df.dropna(subset=['PO Created By'])

    # Normalize fields for reliable comparison
    # pr_df['PR Number'] = pr_df['PR Number'].astype(str).str.strip()
    # po_df['PR Number'] = po_df['PR Number'].astype(str).str.strip()
    pr_df['PR Created By'] = pr_df['PR Created By'].astype(str).str.strip().str.lower()
    po_df['PO Created By'] = po_df['PO Created By'].astype(str).str.strip().str.lower()

    # Merge on PR Number
    merged_df = pd.merge(po_df, pr_df, on='PR Number', how='inner')

    # Filter where both PR and PO were created by same user
    same_user_df = merged_df[merged_df['PO Created By'] == merged_df['PR Created By']].copy()

    return same_user_df[['PO Number', 'PO Creation Date', 'PO Created By', 'PR Number', 'PR Creation Date', 'PR Created By']]


# Category: Purchase Order (PO) vs Purchase Requitition (PR)
# Bot: PR vs PO Item Code Mismatch
def detect_item_code_mismatch(
    pr_df,
    po_df,
    pr_pr_number_col='PR Number',
    pr_item_code_col='Item Code',
    po_pr_number_col='PR Number',
    item_col_col='Item Code',
    filter_pr_to_po_references=True
):
    """
    Detects POs where the combination of PR Number + Item Code does not exist in the PR data.
    This indicates the PO was raised for an item not originally requested.

    Parameters:
        pr_df (pd.DataFrame): Purchase Requisition data
        po_df (pd.DataFrame): Purchase Order data
        pr_pr_number_col (str): Column name for PR Number in PR DataFrame
        pr_item_code_col (str): Column name for Item Code in PR DataFrame
        po_pr_number_col (str): Column name for PR Number in PO DataFrame
        item_col_col (str): Column name for Item Code in PO DataFrame
        filter_pr_to_po_references (bool): Whether to filter PRs to only those referenced in POs (default: True)

    Returns:
        pd.DataFrame: PO rows with item code mismatches, with a mismatch flag added.
                     Returns empty DataFrame if no mismatches found.
                     
    Raises:
        ValueError: If required columns are missing in input DataFrames
    """
    # Input validation
    pr_required = [pr_pr_number_col, pr_item_code_col]
    po_required = [po_pr_number_col, item_col_col]
    
    pr_missing = [col for col in pr_required if col not in pr_df.columns]
    po_missing = [col for col in po_required if col not in po_df.columns]
    
    if pr_missing:
        raise ValueError(f"PR DataFrame missing required columns: {pr_missing}")
    if po_missing:
        raise ValueError(f"PO DataFrame missing required columns: {po_missing}")

    # Create copies to avoid modifying original DataFrames
    pr_clean = pr_df.copy()
    po_clean = po_df.copy()

    # Drop nulls from required fields
    pr_clean = pr_clean.dropna(subset=pr_required)
    po_clean = po_clean.dropna(subset=po_required)

    # Normalize and clean columns
    def clean_column(series):
        return (series.astype(str)
                .str.strip()
                .str.upper()
                .str.replace(r'\.0$', '', regex=True))

    for col in pr_required:
        pr_clean[col] = clean_column(pr_clean[col])
    
    for col in po_required:
        po_clean[col] = clean_column(po_clean[col])

    # Optional: Filter PRs to only those referenced in POs
    if filter_pr_to_po_references:
        referenced_prs = po_clean[po_pr_number_col].unique()
        pr_clean = pr_clean[pr_clean[pr_pr_number_col].isin(referenced_prs)].copy()

    # Create join keys
    pr_clean['Join Key'] = pr_clean[pr_pr_number_col] + '_' + pr_clean[pr_item_code_col]
    po_clean['Join Key'] = po_clean[po_pr_number_col] + '_' + po_clean[item_col_col]

    # Create set of valid PR-Item combinations for faster lookup
    valid_combinations = set(pr_clean['Join Key'])

    # Filter PO rows where Join Key is NOT found in PR
    unmatched_mask = ~po_clean['Join Key'].isin(valid_combinations)
    unmatched_po_df = po_clean[unmatched_mask].copy()
    
    return unmatched_po_df.drop(['Join Key'], axis =1)



# Category: Purchase Order (PO)
# Bot: Backdated Approvals PO
def detect_backdated_po_approvals(po_df, calendar_df=None):

    """
    Detects backdated approvals and approvals on non-working days.
    
    Parameters:
        po_df (pd.DataFrame): PO data with at least ['PO No', 'Creation Date', 'Approval Date', 'Approver'].
        calendar_df (pd.DataFrame, optional): Calendar data with ['Date', 'IsWorkingDay'] (Y/N).
    
    Returns:
        pd.DataFrame: Filtered and flagged approval anomalies.
    """

    # Step 1: Standardize column names and convert to datetime
    po_df['Creation Date'] = pd.to_datetime(po_df['Creation Date'])
    po_df['Approval Date'] = pd.to_datetime(po_df['Approval Date'])

    # Step 2: Flags
    po_df['Backdated?'] = po_df['Approval Date'] < po_df['Creation Date']
    
    # Step 3: Flag weekends and holidays
    po_df['DayOfWeek'] = po_df['Approval Date'].dt.day_name()
    po_df['Weekend?'] = po_df['Approval Date'].dt.weekday >= 5  # Saturday=5, Sunday=6
    po_df['Holiday?'] = False

    if calendar_df is not None:
        calendar_df['Date'] = pd.to_datetime(calendar_df['Date'])
        calendar_df = calendar_df[['Date', 'IsWorkingDay']]
        po_df = pd.merge(po_df, calendar_df, how='left', left_on='Approval Date', right_on='Date')
        po_df['Holiday?'] = po_df['IsWorkingDay'].str.upper() == 'N'
        po_df.drop(columns=['Date', 'IsWorkingDay'], inplace=True)

    # Step 4: Combine flags
    po_df['Weekend/Holiday?'] = po_df.apply(
        lambda row: row['Weekend?'] or row['Holiday?'], axis=1
    )

    # Step 5: Notes
    def make_notes(row):
        if row['Backdated?']:
            return 'Backdated'
        elif row['Weekend/Holiday?']:
            return f"Holiday approval ({row['DayOfWeek']})"
        return ''

    po_df['Notes'] = po_df.apply(make_notes, axis=1)

    # Step 6: Filter only flagged records
    flagged_df = po_df[po_df['Backdated?'] | po_df['Weekend/Holiday?']].copy()

    # # Replace bool with ✅/❌
    # flagged_df['Backdated?'] = flagged_df['Backdated?'].map({True: '✅', False: '❌'})
    # flagged_df['Weekend/Holiday?'] = flagged_df['Weekend/Holiday?'].map({True: '✅', False: '❌'})

    return flagged_df



# Category: Purchase Requitition (PR)
# Bot: Backdated Approval PR
def detect_backdated_pr_approvals(pr_df, calendar_df=None):

    """
    Detects backdated PR approvals and approvals on non-working days.

    Parameters:
        pr_df (pd.DataFrame): PR data with ['PR No', 'Creation Date', 'Approval Date', 'Approver'].
        calendar_df (pd.DataFrame, optional): Calendar data with ['Date', 'IsWorkingDay'] (Y/N).

    Returns:
        pd.DataFrame: Filtered PRs with flagged approval issues.
    """

    # Step 1: Convert dates
    pr_df['Creation Date'] = pd.to_datetime(pr_df['Creation Date'])
    pr_df['Approval Date'] = pd.to_datetime(pr_df['Approval Date'])

    # Step 2: Flag backdated approvals
    pr_df['Backdated?'] = pr_df['Approval Date'] < pr_df['Creation Date']

    # Step 3: Weekend and holiday checks
    pr_df['DayOfWeek'] = pr_df['Approval Date'].dt.day_name()
    pr_df['Weekend?'] = pr_df['Approval Date'].dt.weekday >= 5
    pr_df['Holiday?'] = False

    if calendar_df is not None:
        calendar_df['Date'] = pd.to_datetime(calendar_df['Date'])
        calendar_df = calendar_df[['Date', 'IsWorkingDay']]
        pr_df = pd.merge(pr_df, calendar_df, how='left', left_on='Approval Date', right_on='Date')
        pr_df['Holiday?'] = pr_df['IsWorkingDay'].str.upper() == 'N'
        pr_df.drop(columns=['Date', 'IsWorkingDay'], inplace=True)

    pr_df['Weekend/Holiday?'] = pr_df.apply(lambda r: r['Weekend?'] or r['Holiday?'], axis=1)

    # Step 4: Add notes
    def make_notes(row):
        if row['Backdated?']:
            return 'Backdated'
        elif row['Weekend/Holiday?']:
            return f"Holiday approval ({row['DayOfWeek']})"
        return ''

    pr_df['Notes'] = pr_df.apply(make_notes, axis=1)

    # Step 5: Filter to only anomalies
    flagged = pr_df[pr_df['Backdated?'] | pr_df['Weekend/Holiday?']].copy()

    # flagged['Backdated?'] = flagged['Backdated?'].map({True: '✅', False: '❌'})
    # flagged['Weekend/Holiday?'] = flagged['Weekend/Holiday?'].map({True: '✅', False: '❌'})

    return flagged

# Category: Purchase Order (PO)
# Bot: Backdated PO Creation
def filter_backdated_pos(df, po_date_col, created_date_col):

    """
    Flags POs where the entry (system) date is after the PO/document date, indicating possible backdating.
    Only compares dates (ignores time part if present).
    
    Parameters:
        df (pd.DataFrame): DataFrame containing PO records
        po_date_col (str): Column name for the PO/document date
        created_date_col (str): Column name for the system entry date
    
    Returns:
        pd.DataFrame: Filtered DataFrame with backdated POs flagged
    """
    df = df.copy()
    
    # Convert to datetime
    df['PO Date'] = pd.to_datetime(df['PO Date'], errors='coerce')
    df['Creation Date'] = pd.to_datetime(df['Creation Date'], errors='coerce')
    
    # Compare only dates
    po_dates = df['PO Date'].dt.date
    entry_dates = df['Creation Date'].dt.date
    
    # Filter where entry date > PO date
    flagged_df = df[entry_dates > po_dates].copy()
    flagged_df['Flag'] = 'Backdated PO'
    
    return flagged_df


# Category: Purchase Order (PO)
# Bot: PO Created & Approved by Same User
def filter_same_creator_approver_po(po_df, creater_col='Created By', approved_by_col='Approved By'):
    """
    Filters rows where the same user (case-insensitive, trimmed) created and approved the transaction.
    
    Parameters:
        po_df (pd.DataFrame): DataFrame containing transaction data
        creater_col (str): Column name for 'Created By'
        approved_by_col (str): Column name for 'Approved By'
    
    Returns:
        pd.DataFrame: Filtered DataFrame where creator and approver are the same
    """
    # Clean both columns: strip and lowercase
    po_df['__created_clean__'] = po_df[creater_col].astype(str).str.strip().str.lower()
    po_df['__approved_clean__'] = po_df[approved_by_col].astype(str).str.strip().str.lower()

    # Filter matching entries
    filtered_po_df = po_df[po_df['__created_clean__'] == po_df['__approved_clean__']].copy()
    filtered_po_df['Flag'] = 'Same Creator & Approver'

    # Clean up helper columns
    return filtered_po_df.drop(columns=['__created_clean__', '__approved_clean__'])


# Category: Purchase Order (PO) vs Purchase Requitition (PR)
# Bot: PR Created & Approved by Same User
def filter_same_creator_approver_pr(pr_df, creater_col, approved_by_col):

    """
    Filters rows where the same user (case-insensitive, trimmed) created and approved the transaction.
    
    Parameters:
        pr_df (pd.DataFrame): DataFrame containing transaction data
        creater_col (str): Column name for 'Created By'
        approved_by_col (str): Column name for 'Approved By'
    
    Returns:
        pd.DataFrame: Filtered DataFrame where creator and approver are the same
    """
    # Clean both columns: strip and lowercase
    pr_df['__created_clean__'] = pr_df["Created By"].astype(str).str.strip().str.lower()
    pr_df['__approved_clean__'] = pr_df["Approved By"].astype(str).str.strip().str.lower()

    # Filter matching entries
    filtered_pr_df = pr_df[pr_df['__created_clean__'] == pr_df['__approved_clean__']].copy()
    filtered_pr_df['Flag'] = 'Same Creator & Approver'

    # Clean up helper columns
    return filtered_pr_df.drop(columns=['__created_clean__', '__approved_clean__'])


# Category: Purchase Order (PO)
# Bot: PO Created Outside Business Hours
#__________________________________________________________________________________
# Purchase Order (PO)
import pandas as pd
 
def flag_pos_outside_business_hours(po_df, timestamp_col, weekends={5, 6}, holidays=None, office_start=9, office_end=18):
# PO Created Outside Business Hours
    """
    Flags POs created during weekends, holidays, or outside standard business hours.
    Ignores rows where time component is missing.
   
    Parameters:
        po_df (pd.DataFrame): DataFrame containing PO records.
        timestamp_col (str): Column name for PO creation timestamp.
        weekends (set): Set of integers indicating weekend days (default: Saturday & Sunday).
        holidays (set or list): Optional list of holiday dates (as datetime.date).
        office_start (int): Business start hour (inclusive).
        office_end (int): Business end hour (exclusive).
   
    Returns:
        pd.DataFrame: Flagged POs with Notes on Weekend/Holiday/After Hours issues.
    """
    po_df = po_df.copy()
    po_df['Creation Date'] = pd.to_datetime(po_df['Creation Date'], errors='coerce')
 
    po_df['DayOfWeek'] = po_df['Creation Date'].dt.weekday
    po_df['Date'] = po_df['Creation Date'].dt.date
    po_df['Hour'] = po_df['Creation Date'].dt.hour
    po_df['Minute'] = po_df['Creation Date'].dt.minute
    po_df['Second'] = po_df['Creation Date'].dt.second
 
    # Determine if time info is available (not 00:00:00)
    po_df['HasTime'] = ~((po_df['Hour'] == 0) & (po_df['Minute'] == 0) & (po_df['Second'] == 0))
 
    # Weekend flag
    po_df['Weekend?'] = po_df['DayOfWeek'].isin(weekends)
 
    # Holiday flag
    if holidays:
        holidays = set(pd.to_datetime(holidays).date)
        po_df['Holiday?'] = po_df['Date'].isin(holidays)
    else:
        po_df['Holiday?'] = False
 
    # Outside business hours (only if time info exists)
    po_df['Outside Hours?'] = po_df.apply(
        lambda row: not (office_start <= row['Hour'] < office_end) if row['HasTime'] else False,
        axis=1
    )
 
    # Notes column
    def get_notes(row):
        notes = []
        if row['Weekend?']:
            notes.append("Weekend")
        if row['Holiday?']:
            notes.append("Holiday")
        if row['HasTime'] and row['Outside Hours?']:
            notes.append("After Hours")
        return ", ".join(notes)
 
    po_df['Notes'] = po_df.apply(get_notes, axis=1)
 
    # Filter only rows that have time and are actually flagged
    flagged = po_df[(po_df['HasTime']) & (po_df['Notes'] != '')].copy()
 
    return flagged.drop(columns=['Hour', 'Minute', 'Second', 'HasTime'])
 



# Category: Purchase Requitition (PR)
# Bot: PR Created Outside Business Hours
import pandas as pd
 
def flag_prs_outside_business_hours(pr_df, timestamp_col, weekends={5, 6}, holidays=None, office_start=9, office_end=18):

    """
    Flags prs created during weekends, holidays, or outside standard business hours.
    Ignores rows where time comprnent is missing.
   
    Parameters:
        pr_df (pd.DataFrame): DataFrame containing pr records.
        timestamp_col (str): Column name for pr creation timestamp.
        weekends (set): Set of integers indicating weekend days (default: Saturday & Sunday).
        holidays (set or list): Optional list of holiday dates (as datetime.date).
        office_start (int): Business start hour (inclusive).
        office_end (int): Business end hour (exclusive).
   
    Returns:
        pd.DataFrame: Flagged prs with Notes on Weekend/Holiday/After Hours issues.
    """
    pr_df = pr_df.copy()
    pr_df['Creation Date'] = pd.to_datetime(pr_df['Creation Date'], errors='coerce')
 
    pr_df['DayOfWeek'] = pr_df['Creation Date'].dt.weekday
    pr_df['Date'] = pr_df['Creation Date'].dt.date
    pr_df['Hour'] = pr_df['Creation Date'].dt.hour
    pr_df['Minute'] = pr_df['Creation Date'].dt.minute
    pr_df['Second'] = pr_df['Creation Date'].dt.second
 
    # Determine if time info is available (not 00:00:00)
    pr_df['HasTime'] = ~((pr_df['Hour'] == 0) & (pr_df['Minute'] == 0) & (pr_df['Second'] == 0))
 
    # Weekend flag
    pr_df['Weekend?'] = pr_df['DayOfWeek'].isin(weekends)
 
    # Holiday flag
    if holidays:
        holidays = set(pd.to_datetime(holidays).date)
        pr_df['Holiday?'] = pr_df['Date'].isin(holidays)
    else:
        pr_df['Holiday?'] = False
 
    # Outside business hours (only if time info exists)
    pr_df['Outside Hours?'] = pr_df.apply(
        lambda row: not (office_start <= row['Hour'] < office_end) if row['HasTime'] else False,
        axis=1
    )
 
    # Notes column
    def get_notes(row):
        notes = []
        if row['Weekend?']:
            notes.append("Weekend")
        if row['Holiday?']:
            notes.append("Holiday")
        if row['HasTime'] and row['Outside Hours?']:
            notes.append("After Hours")
        return ", ".join(notes)
 
    pr_df['Notes'] = pr_df.apply(get_notes, axis=1)
 
    # Filter only rows that have time and are actually flagged
    flagged = pr_df[(pr_df['HasTime']) & (pr_df['Notes'] != '')].copy()
 
    return flagged.drop(columns=['Hour', 'Minute', 'Second', 'HasTime'])



# Category: Purchase Requitition (PR)
# Bot: PR with Missing Supporting Fields
def get_prs_with_missing_supporting_fields_pr_df(pr_df):

    """
    Identify POs where any of the specified supporting fields are missing or blank.
    Ignores fields not present in the DataFrame.

    Parameters:
        pr_df (pd.DataFrame): Input PO data.
        required_fields_pr (list): Fields to validate.

    Returns:
        pd.DataFrame: Rows with at least one missing required field and a note column listing them.
    """
    required_fields_pr = ['Required Date', 'Required Quantity', 'Cost Center', 'Item Code', 'Item Description', 'Created by', 'Approved by']

    pr_df = pr_df.copy()
    pr_df.columns = pr_df.columns.str.strip()

    # Keep only required fields that actually exist in the DataFrame
    existing_fields = [field for field in required_fields_pr if field in pr_df.columns]

    # Normalize relevant columns
    for col in existing_fields:
        pr_df[col] = pr_df[col].astype(str).str.strip().replace(["nan", "NaT"], "")

    # Identify missing fields row-wise
    def find_missing_fields(row):
        return [field for field in existing_fields if row[field] == '']

    # Apply function and filter
    pr_df['Missing Fields'] = pr_df.apply(lambda row: ', '.join(find_missing_fields(row)), axis=1)
    filtered_pr_df = pr_df[pr_df['Missing Fields'] != '']
    return filtered_pr_df


# Category: Purchase Order (PO)
# Bot: PO with Missing Supporting Fields
def get_pos_with_missing_supporting_fields_po_df(po_df):

    """
    Identify POs where any of the specified supporting fields are missing or blank.
    Ignores fields not present in the DataFrame.

    Parameters:
        po_df (pd.DataFrame): Input PO data.
        required_fields_po (list): Fields to validate.

    Returns:
        pd.DataFrame: Rows with at least one missing required field and a note column listing them.
    """
    po_df = po_df.copy()
    po_df.columns = po_df.columns.str.strip()
    
    required_fields_po = ['Delivery Date', 'Vendor Code', 'Quantity', 'Item Code', 'Unit Price', 'Created by', 'Approved by']

    # Keep only required fields that actually exist in the DataFrame
    existing_fields = [field for field in required_fields_po if field in po_df.columns]

    # Normalize relevant columns
    for col in existing_fields:
        po_df[col] = po_df[col].astype(str).str.strip().replace(["nan", "NaT"], "")

    # Identify missing fields row-wise
    def find_missing_fields(row):
        return [field for field in existing_fields if row[field] == '']

    # Apply function and filter
    po_df['Missing Fields'] = po_df.apply(lambda row: ', '.join(find_missing_fields(row)), axis=1)
    filtered_po_df = po_df[po_df['Missing Fields'] != '']

    return filtered_po_df



# Category: Purchase Order (PO)
# Bot: PO Price Spikes
def flag_po_price_spikes(po_df, threshold=0.10):

    """
    Flags POs with price spikes compared to previous POs for the same item.
    
    Parameters:
        po_df (pd.DataFrame): PO line-item history.
        item_col (str): Column name for material/item ID.
        price_col (str): Column name for unit price.
        date_col (str): Column name for PO date (datetime or string).
        threshold (float): % price change threshold (e.g., 0.1 for 10%).
    
    Returns:
        pd.DataFrame: Subset with price spikes flagged, % change, and notes.
    """
    df = po_df.copy()

    # Parse date and sort for proper history analysis
    df['Creation Date'] = pd.to_datetime(df['Creation Date'], errors='coerce')
    df.sort_values(by=["Item Code", 'Creation Date'], inplace=True)

    # Group by material and shift previous price & date
    df['Previous_Price'] = df.groupby("Item Code")['Unit Price'].shift(1)
    df['Previous_Date'] = df.groupby("Item Code")['Creation Date'].shift(1)

    # Calculate % change
    df['%Change'] = (df['Unit Price'] - df['Previous_Price']) / df['Previous_Price']

    # Handle edge cases
    df['%Change'] = df['%Change'].replace([float('inf'), -float('inf')], pd.NA)

    # Flag if % change ≥ threshold
    df['Price_Spike_Flag'] = df['%Change'].abs() >= threshold

    # Add notes column
    def get_notes(row):
        if pd.isna(row['Previous_Price']):
            return "No prior price"
        elif row['Price_Spike_Flag']:
            return f"Price spike of {row['%Change']:.1%}"
        return ""

    df['Notes'] = df.apply(get_notes, axis=1)

    # Return flagged rows only
    flagged = df[df['Price_Spike_Flag'] == True].copy()
    return flagged[['Item Code', 'Creation Date', 'Unit Price','Previous_Date', 'Previous_Price', '%Change', 'Notes']]


# Category: Purchase Order (PO)
# BOT: Potential PO Splitting
def detect_po_splitting(po_df,
                        vendor_col='Vendor Code',
                        created_date_col='Creation Date',
                        po_amount_col='GTotal',
                        bsart_col=None,
                        exclude_bsart_value='FO',
                        po_number_col='purchaseorder_number',
                        item_col='Item Code',
                        creator_col='Created By',
                        approval_threshold=100000):
    """
    Enhanced PO splitting detection that properly handles multiple items per PO.
   
    This function identifies potential cases where a single large purchase order may have been
    split into multiple smaller POs to avoid approval thresholds. It analyzes by vendor and date,
    checking for multiple POs that collectively exceed the threshold and share common characteristics.
   
    Args:
        po_df: DataFrame containing purchase order data
        vendor_col: Column name for vendor identifier
        created_date_col: Column name for PO creation date
        po_amount_col: Column name for PO amount
        bsart_col: Optional column name for PO type
        exclude_bsart_value: PO type value to exclude (default 'FO')
        po_number_col: Column name for PO number
        item_col: Column name for item code (optional)
        creator_col: Column name for creator identifier (optional)
        approval_threshold: Amount threshold for approval (default 100000)
   
    Returns:
        DataFrame with flagged potential PO splits containing:
        - All original columns
        - 'Notes' column with detected patterns
        - 'PO_Group_ID' for grouping related POs
    """
   
    # Handle empty DataFrame input
    if po_df.empty:
        return pd.DataFrame(columns=po_df.columns.tolist() + ['Notes', 'PO_Group_ID'])
   
    # Create working copy and standardize column names
    df = po_df.copy()
    df.columns = df.columns.str.strip()
   
    # Validate required columns exist (only vendor, date, amount, and PO number are required)
    required_cols = {vendor_col, created_date_col, po_amount_col, po_number_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
   
    # Convert and validate date column
    df[created_date_col] = pd.to_datetime(df[created_date_col], errors='coerce')
    if df[created_date_col].isna().any():
        raise ValueError(f"Invalid date values found in {created_date_col}")
   
    # Filter out excluded PO types if specified
    if bsart_col and bsart_col in df.columns:
        df = df[~df[bsart_col].str.upper().eq(exclude_bsart_value.upper())].copy()
   
    # Normalize dates (remove time component)
    df['PO_Date'] = df[created_date_col].dt.normalize()
   
    # Initialize variables for item analysis (if item_col exists)
    item_analysis = item_col in df.columns
    if item_analysis:
        # Create a mapping of PO numbers to their item sets
        po_items = (
            df.groupby([vendor_col, 'PO_Date', po_number_col])[item_col]
            .apply(set)
            .reset_index(name='Items')
        )
       
        # For each vendor-date group, get the union of all items across POs
        all_items_per_group = (
            po_items.groupby([vendor_col, 'PO_Date'])['Items']
            .apply(lambda x: set.union(*x))
            .reset_index(name='All_Items')
        )
   
    # Prepare aggregation dictionary for summary statistics
    agg_dict = {
        'PO_Count': (po_number_col, 'nunique'),  # Number of unique POs
        'Total_Amount': (po_amount_col, 'sum'),  # Sum of all PO amounts
        'PO_Numbers': (po_number_col, lambda x: sorted(set(x)))  # List of PO numbers
    }
   
    # Add creator uniqueness if creator_col exists
    if creator_col in df.columns:
        agg_dict['Unique_Creators'] = (creator_col, 'nunique')
   
    # Create summary statistics for each vendor-date group
    summary = (
        df.groupby([vendor_col, 'PO_Date'])
        .agg(**agg_dict)
        .reset_index()
    )
   
    # Merge the item information if available
    if item_analysis:
        summary = summary.merge(all_items_per_group, on=[vendor_col, 'PO_Date'], how='left')
        summary = summary.merge(
            po_items.groupby([vendor_col, 'PO_Date'])['Items']
            .apply(list)
            .reset_index(name='PO_Items_List'),
            on=[vendor_col, 'PO_Date'],
            how='left'
        )
   
    # Flag groups that meet the splitting criteria:
    # 1. Multiple POs (>= 2) on the same day to same vendor
    # 2. Combined amount exceeds approval threshold
    flagged_summary = summary[
        (summary['PO_Count'] >= 2) &
        (summary['Total_Amount'] >= approval_threshold)
    ].copy()
   
    # Return empty DataFrame if no suspicious groups found
    if flagged_summary.empty:
        return pd.DataFrame(columns=df.columns.tolist() + ['Notes', 'PO_Group_ID'])
 
    # Assign unique group IDs to each flagged vendor-date group
    flagged_summary = flagged_summary.reset_index(drop=True)
    flagged_summary['PO_Group_ID'] = 'GROUP_' + (flagged_summary.index + 1).astype(str)
 
    def generate_notes(row):
        """Generate descriptive notes about potential splitting patterns."""
        notes = []
       
        # Check if all POs were created by same person (if creator_col exists)
        if creator_col in df.columns and 'Unique_Creators' in row and row['Unique_Creators'] == 1:
            notes.append("Same Creator")
 
        # Check item patterns if item_col exists and item analysis was performed
        if item_analysis and 'PO_Items_List' in row and isinstance(row['PO_Items_List'], list):
            # Flatten all items from all POs in this group
            all_items = []
            for item_set in row['PO_Items_List']:
                all_items.extend(list(item_set))
 
            # First Check if all POs contain exactly the same items
            first_set = set(row['PO_Items_List'][0])
            if all(set(s) == first_set for s in row['PO_Items_List']):
                notes.append("Same Items")
            # Only Check for overlapping items across POs if not Identical
            elif len(all_items) > len(set(all_items)):
                notes.append("Common Items")
 
        return " & ".join(sorted(set(notes))) if notes else ""
   
    # Apply note generation to each flagged group
    flagged_summary['Notes'] = flagged_summary.apply(generate_notes, axis=1)
 
    # Merge the flags and notes back to the original PO records
    result_df = (
        df.merge(
            flagged_summary[[vendor_col, 'PO_Date', 'Notes', 'PO_Group_ID']],
            on=[vendor_col, 'PO_Date'],
            how='inner'
        )
        .drop(columns=['PO_Date'])  # Remove temporary date column
        .sort_values([vendor_col, created_date_col])  # Sort by vendor and creation date
        .reset_index(drop=True)
    )
   
    return result_df
# Category: Purchase Order (PO)
# BOT: PO Without Approver
def flag_missing_approver_ids_po(po_df, 
                                  approver_col='Approved By',
                                  approve_type_col=None,
                                  auto_approved_value=None):
    """
    Flags POs where the Approved By is missing or blank,
    optionally excluding POs marked as auto-approved.

    Parameters:
        po_df (pd.DataFrame): DataFrame containing PO recostrds.
        approver_col (str): Column name for the Approved By field.
        approve_type_col (str or None): Column indicating type of approval (optional).
        auto_approved_value (str or None): Value in approve_type_col that indicates auto-approval (optional).

    Returns:
        pd.DataFrame: Filtered DataFrame with missing approver information and a 'Flag' column.
    """
    po_df = po_df.copy()
    po_df.columns = po_df.columns.str.strip()

    # Normalize approver field
    po_df[approver_col] = po_df[approver_col].astype(str).str.strip().fillna("").replace("nan", "", regex=False)

    # Flag rows with missing or empty Approved By
    flagged_df = po_df[po_df[approver_col] == ''].copy()

    # If approval type filtering is needed
    if approve_type_col and auto_approved_value is not None and approve_type_col in flagged_df.columns:
        flagged_df = flagged_df[~flagged_df[approve_type_col].astype(str).str.strip().eq(str(auto_approved_value).strip())]

    flagged_df['Flag'] = 'Missing Approved By'
    return flagged_df


# Category: Purchase Requitition (PR)
# BOT: PR Without Approver
def flag_missing_approver_ids_pr(pr_df, 
                                  approver_col='Approved By',
                                  approve_type_col=None,
                                  auto_approved_value=None):
    """
    Flags prs where the Approved By is missing or blank,
    optionally excluding prs marked as auto-approved.

    Parameters:
        pr_df (pd.DataFrame): DataFrame containing pr records.
        approver_col (str): Column name for the Approved By field.
        approve_type_col (str or None): Column indicating type of approval (optional).
        auto_approved_value (str or None): Value in approve_type_col that indicates auto-approval (optional).

    Returns:
        pd.DataFrame: Filtered DataFrame with missing approver information and a 'Flag' column.
    """
    pr_df = pr_df.copy()
    pr_df.columns = pr_df.columns.str.strip()

    # Normalize approver field
    pr_df[approver_col] = pr_df[approver_col].astype(str).str.strip().fillna("").replace("nan", "", regex=False)

    # Flag rows with missing or empty Approved By
    flagged_df = pr_df[pr_df[approver_col] == ''].copy()

    # If approval type filtering is needed
    if approve_type_col and auto_approved_value is not None and approve_type_col in flagged_df.columns:
        flagged_df = flagged_df[~flagged_df[approve_type_col].astype(str).str.strip().eq(str(auto_approved_value).strip())]

    flagged_df['Flag'] = 'Missing Approved By'
    return flagged_df
