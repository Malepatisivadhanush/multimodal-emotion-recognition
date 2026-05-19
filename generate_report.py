from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib import styles

doc = SimpleDocTemplate("Project_Report.pdf")

styleSheet = styles.getSampleStyleSheet()

story=[]

with open("report_content.txt","r",encoding="utf-8") as f:
    content=f.readlines()

for line in content:

    line=line.strip()

    if line=="":

        story.append(
            Spacer(1,12)
        )

    else:

        story.append(
            Paragraph(
                line,
                styleSheet['BodyText']
            )
        )

doc.build(story)

print("PDF generated successfully")