import tkinter as tk
from PIL import Image, ImageOps
from back import response
import io

x1, y1 = 0, 0
brush_size = 3  # Initial brush size

root = tk.Tk()
root.title("Handwriting Checker")

# Customizable colours
background = '#352F44'
fontColour1 = '#F5F5F5'
fontColourButton = '#18122B'
buttonColour = '#B9B4C7'
buttonPressed = "#635985"
buttonAction1 = '#4D4C7D'
fontText = 'Kristen ITC'
fontTextLetters = 'Tahoma'

# Set window size to full screen
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{screen_width}x{screen_height}")
root.configure(background=background)

def show_new_page(letter):
    # Creates a new window (page)
    new_page = tk.Toplevel(root)
    new_page.configure(background=background )
    new_page.title(f"You selected: {letter}")
    new_page.geometry(f"{screen_width}x{screen_height}")

    label1 = tk.Label(new_page, background=background , text=f"You selected: {letter}", fg=fontColour1, font=(fontText, 25, 'bold'))
    label1.pack(padx=20, pady=25)

    label2 = tk.Label(new_page, background=background , text="Try your best! Click 'done' once you're finished.", fg=fontColour1, font=(fontText, 20))
    label2.pack(padx=20, pady=10)

    def canvas_to_image(canvas, filename='canvas_image.png'):
        # Save postscript image string
        ps = canvas.postscript(colormode='color')

        # Use PIL to convert to PNG
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        img.save(filename)

    def rotate_and_invert_image(image_path):
        # Open the image file
        with Image.open(image_path) as image:
            inverted_image = ImageOps.invert(image)
            # Rotate the image 90 degrees counterclockwise
            rotated_image = inverted_image.rotate(-90, expand=True)
            # Invert the image horizontally
            inverted_image = rotated_image.transpose(Image.FLIP_LEFT_RIGHT)

            # Save the transformed image as 'canvas_image.png'
            inverted_image.save("canvas_image.png")

    def get_coords(event):
        global x1, y1
        x1, y1 = event.x, event.y

    def draw(event):
        global x1, y1
        canvas.create_line((x1, y1, event.x, event.y), fill='black', width=brush_size)
        x1, y1 = event.x, event.y

    def clear_canvas():
        canvas.delete("all")  # Clears all items on the canvas

    def increase_brush():
        global brush_size
        brush_size += 1

    def decrease_brush():
        global brush_size
        if brush_size > 1:
            brush_size -= 1

    def result():
        #Convert the canvas to an image
        canvas_to_image(canvas)
        #Rotate and invert the image
        rotate_and_invert_image("canvas_image.png")
        response_result = response("canvas_image.png", letter)
        if response_result[0] == letter and response_result[1] >= 0.5:
            print('Good job')
        else:
            print('You got it twisted')


    canvas = tk.Canvas(new_page, bg='white', width=400, height=400)
    canvas.pack(anchor='center')

    canvas.bind("<Button-1>", get_coords)
    canvas.bind("<B1-Motion>", draw)

    button_frame_combined = tk.Frame(new_page, bg=background , padx=20, pady=20)
    button_frame_combined.pack(pady=10)

    # Create buttons within the combined frame

    btn_back = tk.Button(button_frame_combined, text="Back", bg ='#818FB4', fg=fontColourButton, activebackground="#435585", activeforeground=fontColourButton, font=(fontText, 13), command=new_page.destroy)
    btn_clear = tk.Button(button_frame_combined, text="Clear", bg ='#FF6969', fg=fontColourButton, activebackground="#CE5A67", activeforeground=fontColourButton, font=(fontText, 13), command=clear_canvas)
    btn_done = tk.Button(
        button_frame_combined,
        text="Done",
        bg='#A2C579',
        fg='white',
        activebackground="#748E63",
        activeforeground="white",
        font=('Comic Sans MS', 12),
        command=result  #call the result function directly
    )
    btn_increase = tk.Button(button_frame_combined, text="Increase Brush", font=(fontText, 13), command=increase_brush)
    btn_decrease = tk.Button(button_frame_combined, text="Decrease Brush", font=(fontText, 13), command=decrease_brush)

    btn_back.pack(side="left", padx=10)  # Position the button within the frame, with some space between them
    btn_clear.pack(side="left", padx=10)
    btn_done.pack(side="left", padx=10)
    btn_increase.pack(side="left", padx=10)
    btn_decrease.pack(side="left", padx=10)

# Main page labels
label = tk.Label(root, text="Handwriting Checker", bg=background , fg=fontColour1, font=(fontText, 50, 'bold'))
label.pack(padx=20, pady=30)

label = tk.Label(root, text="Pick a letter or number.", bg=background , fg=fontColour1, font=(fontText, 30))
label.pack(padx=20, pady=15)

buttonframe = tk.Frame(root, bg=background )  # Set the background color of the buttonframe
buttonframe.pack()

all_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890'

def create_buttons(buttonframe, characters, start_row):
    for i, char in enumerate(characters):
        row = i // 13 + start_row
        column = i % 13
        button = tk.Button(
            buttonframe,
            width=2,
            text=char,
            bd=3,
            bg='#A2C579',
            fg='white',
            activebackground="#748E63",
            activeforeground="white",
            font=('Comic Sans MS', 21),
            command=lambda char=char: show_new_page(char)
        )
        button.grid(row=row, column=column)


# Define the function to add a gap between button sections
def add_row_gap_with_colour(row_index, colour=background , num_columns=13):
  label = tk.Label(buttonframe, height=2, width=num_columns * 3, bg=colour)  # Create a coloured label
  label.grid(row=row_index, column=0, columnspan=num_columns, sticky="ew")  # Place the label in the specified row

add_row_gap_with_colour(0, colour=background )  # Gap with a specific colour after the first section
create_buttons(buttonframe, all_characters[:26], 0)  # Uppercase characters

add_row_gap_with_colour(2, colour=background )  # Gap with a specific colour after the second section
create_buttons(buttonframe, all_characters[26:52], 3)  # Lowercase characters

add_row_gap_with_colour(5, colour=background )  # Gap with a specific colour after the third section
create_buttons(buttonframe, all_characters[52:], 6)  # Numbers


root.mainloop()
