# Exploding Ketchup's innovation project
## Our FLL innovation project's code
### What it does:
- It takes a picture from your computer webcam, and then analyses the emotions of the person in the photo. Then, it generates an image using AI based on the person's emotions, as well as other information that they provide including their favourite hobbies and their favourite food.
- Specifically, we take the picture using OpenCV (Python bindings, https://opencv.org/). Then we use Deepface (https://github.com/serengil/deepface) to analyse the emotions of the person in the image. We then pass the emotions to ChatGPT's API (gpt-3.5-turbo) and you can see the prompt we use as Chatgptprompt.txt.
- After this, ChatGPT generates a prompt for Stable Diffusion, an AI text-to-image model. We used Dream Studio's API (https://dreamstudio.ai/) to generate the final image.
- You can enter your API keys into the .env file.

### How to run it:
- Tested and working on Python 3.11.3, will most likely work for other versions.
- Enter your API keys into the .env file.
- Install the following modules with pip: `pip install openai deepface opencv-python`
- Run `opencvtest.py` and you should see a feed from your webcam that has green rectangle over your face
- Press `s` to save when you are happy with it and `q` to quit the camera app.
- Then run `mainprogram.py`  and follow the instructions displayed on the menu
- Some of the modules will take a while to load and will print some incomprehensible blabber - don't worry about this.
- When the image gets generated, it will be saved in the project folder.

# Have fun!