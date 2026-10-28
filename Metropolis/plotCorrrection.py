from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Open the PNG file
def main():
    image = Image.open("D:\\PhD_\\sn-article-template\\Portfolio_test_plot_p.png")

    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Define the font and font size
    font = ImageFont.truetype("arial.ttf", 20)  # Change the font and size as needed

    # Get the current axis limits
    xlim = plt.xlim()
    ylim = plt.ylim()

    # Draw x-axis label
    draw.text((xlim[1] - 100, ylim[0] + 10), "X Axis Label", fill="black", font=font)

    # Draw y-axis label
    draw.text((xlim[0] + 10, ylim[1] - 40), "Y Axis Label", fill="black", font=font)

    # Save the modified image
    image.save("modified_image.png")


if __name__ == '__main__':
    main()