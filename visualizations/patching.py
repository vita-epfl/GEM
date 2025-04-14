from PIL import Image, ImageDraw


def draw_grid_on_image(input_image_path, output_image_path, ncolumns, nrows):
    # Open an image file
    with Image.open(input_image_path) as img:
        # Create a drawing object
        draw = ImageDraw.Draw(img)
        width, height = img.size

        # Calculate the size of each cell in the grid
        cell_width = width / ncolumns
        cell_height = height / nrows

        # Draw vertical lines
        for i in range(1, ncolumns):
            x = i * cell_width
            draw.line([(x, 0), (x, height)], fill="red", width=2)

        # Draw horizontal lines
        for i in range(1, nrows):
            y = i * cell_height
            draw.line([(0, y), (width, y)], fill="red", width=2)

        # Save the result
        img.save(output_image_path)
        print(f"Image saved as {output_image_path}")


# Example usage
draw_grid_on_image("example.png", "example_patches.png", ncolumns=10, nrows=5)
