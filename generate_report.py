from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    PageBreak
)

from reportlab.lib import styles

doc = SimpleDocTemplate("Project_Report.pdf")

styleSheet = styles.getSampleStyleSheet()

story = []

# Read report content
with open("report_content.txt", "r", encoding="utf-8") as f:
    content = f.readlines()

# Add text content
for line in content:

    line = line.strip()

    if line == "":

        story.append(
            Spacer(1, 12)
        )

    else:

        story.append(
            Paragraph(
                line,
                styleSheet['BodyText']
            )
        )

# Add space before images
story.append(Spacer(1, 20))
story.append(PageBreak())

# Add confusion matrix heading
story.append(
    Paragraph(
        "Confusion Matrix Visualization",
        styleSheet['Heading2']
    )
)

story.append(Spacer(1, 12))

# Add confusion matrix image
story.append(
    Image(
        "results/plots/speech_confusion_matrix.png",
        width=400,
        height=300
    )
)

story.append(Spacer(1, 20))

story.append(PageBreak())

# Add t-SNE heading
story.append(
    Paragraph(
        "t-SNE Visualization",
        styleSheet['Heading2']
    )
)

story.append(Spacer(1, 12))

# Add t-SNE image
story.append(
    Image(
        "results/plots/tsne_speech.png",
        width=400,
        height=300
    )
)

story.append(Spacer(1, 20))

# Build PDF
doc.build(story)

print("PDF generated successfully")