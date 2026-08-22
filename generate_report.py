from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from pathlib import Path

output_path = Path(__file__).parent / "CreditWise_Project_Report.pdf"

# Register a common font to avoid issues on Windows
try:
    pdfmetrics.registerFont(TTFont('DejaVuSans', 'C:/Windows/Fonts/arial.ttf'))
except Exception:
    pdfmetrics.registerFont(TTFont('DejaVuSans', 'C:/Windows/Fonts/arial.ttf'))

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleStyle', fontName='DejaVuSans', fontSize=22, leading=28, alignment=TA_CENTER, textColor=colors.HexColor('#1f4e79'), spaceAfter=20))
styles.add(ParagraphStyle(name='SubtitleStyle', fontName='DejaVuSans', fontSize=11, alignment=TA_CENTER, textColor=colors.grey, spaceAfter=20))
styles.add(ParagraphStyle(name='HeadingStyle', fontName='DejaVuSans', fontSize=14, leading=18, textColor=colors.HexColor('#0b5394'), spaceAfter=8, spaceBefore=12))
styles.add(ParagraphStyle(name='BodyStyle', fontName='DejaVuSans', fontSize=10.5, leading=15, alignment=TA_JUSTIFY, spaceAfter=6))
styles.add(ParagraphStyle(name='BulletStyle', fontName='DejaVuSans', fontSize=10.5, leading=14, leftIndent=18, bulletIndent=0, spaceAfter=4))

story = []
story.append(Paragraph("CreditWise Loan Approval Prediction System", styles['TitleStyle']))
story.append(Paragraph("Project Report", styles['SubtitleStyle']))
story.append(Paragraph("Prepared by: Joydeep Singh Kalra", styles['SubtitleStyle']))
story.append(Spacer(1, 12))

story.append(Paragraph("1. Project Overview", styles['HeadingStyle']))
story.append(Paragraph("CreditWise is a machine learning-based loan approval prediction system designed to assist financial institutions in making faster and more consistent decisions. The project uses historical applicant data to predict whether a loan should be approved or rejected.", styles['BodyStyle']))

story.append(Paragraph("2. Problem Statement", styles['HeadingStyle']))
story.append(Paragraph("Manual loan verification is time-consuming and prone to inconsistencies. The goal of this system is to reduce human effort while improving decision consistency and reducing financial risk.", styles['BodyStyle']))

story.append(Paragraph("3. Dataset", styles['HeadingStyle']))
story.append(Paragraph("The project uses a structured dataset containing applicant demographics, financial information, loan details, and the target variable Loan_Approved. The dataset includes both numerical and categorical attributes.", styles['BodyStyle']))

story.append(Paragraph("4. Methodology", styles['HeadingStyle']))
methods = [
    "Loaded and inspected the loan dataset using pandas.",
    "Handled missing values through imputation for both numerical and categorical columns.",
    "Performed exploratory data analysis using visualizations such as pie charts, histograms, and boxplots.",
    "Engineered new features such as squared DTI and Credit Score terms.",
    "Encoded categorical variables and scaled features for modeling.",
    "Trained multiple classifiers and compared their performance.",
]
story.append(ListFlowable([ListItem(Paragraph(item, styles['BulletStyle'])) for item in methods], bulletType='bullet'))

story.append(Paragraph("5. Models and Results", styles['HeadingStyle']))
story.append(Paragraph("The project compared Logistic Regression, K-Nearest Neighbors, and Gaussian Naive Bayes. Logistic Regression produced the best overall performance with strong precision and recall balance.", styles['BodyStyle']))
story.append(Paragraph("Observed metrics: Accuracy 0.875, Precision 0.79, Recall 0.803, and F1-score 0.796.", styles['BodyStyle']))

story.append(Paragraph("6. Streamlit Deployment", styles['HeadingStyle']))
story.append(Paragraph("The project was also deployed as a Streamlit web application so users can input applicant details and receive an instant loan approval prediction through the trained model(Logistic Regression).", styles['BodyStyle']))

story.append(Paragraph("7. Key Insights", styles['HeadingStyle']))
story.append(Paragraph("Credit Score and Debt-to-Income Ratio were identified as more influential predictors than income alone. This highlights the importance of repayment behavior and financial discipline in loan approval decisions.", styles['BodyStyle']))

story.append(Paragraph("8. Conclusion", styles['HeadingStyle']))
story.append(Paragraph("CreditWise demonstrates an end-to-end machine learning workflow for loan risk assessment. It combines data preprocessing, visualization, model training, evaluation, and deployment in a practical and user-friendly format.", styles['BodyStyle']))

story.append(PageBreak())
story.append(Paragraph("Appendix", styles['HeadingStyle']))
story.append(Paragraph("Files included in the project: credit_wise.ipynb, loan_approval_data.csv, app.py, requirements.txt, and this report.", styles['BodyStyle']))

pdf = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=48, leftMargin=48, topMargin=36, bottomMargin=36)
pdf.build(story)
print(f"Report created: {output_path}")
