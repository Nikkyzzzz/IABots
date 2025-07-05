import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
import pandas as pd
from io import BytesIO

# Load email credentials
load_dotenv()
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")


def send_reports_via_email(recipient_email: str, report_dict: dict) -> tuple:
    """
    Combines all reports into one Excel file (multi-sheet) and sends via email.
    :param recipient_email: destination email address
    :param report_dict: dict of {sheet_name: pd.DataFrame}
    :return: (success: bool, message: str)
    """
    try:
        # Create Excel with multiple sheets
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            for sheet_name, df in report_dict.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    df.to_excel(writer, sheet_name=sheet_name[:31], index=False)  # Excel sheet name limit = 31

        buffer.seek(0)
        attachment_data = buffer.read()

        # Create email
        email = EmailMessage()
        email["Subject"] = "📊 Consolidated Procurement Bot Report"
        email["From"] = EMAIL_ADDRESS
        email["To"] = recipient_email
        email.set_content("Please find attached the consolidated procurement report.")

        # Attach file
        email.add_attachment(
            attachment_data,
            maintype="application",
            subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename="Procurement_Report.xlsx"
        )

        # Send email
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.starttls()
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(email)

        return True, "✅ Email with consolidated report sent successfully."

    except Exception as e:
        return False, f"❌ Failed to send email: {e}"


if __name__ == "__main__":
    # For test run
    test_df = pd.DataFrame({"Name": ["Test"], "Value": [123]})
    success, msg = send_reports_via_email(EMAIL_ADDRESS, {"Sample Report": test_df})
    print(msg)
