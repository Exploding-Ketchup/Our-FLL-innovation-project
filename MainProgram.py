from deepface import DeepFace
from openai import OpenAI
from ProjectConstants import *
import json
import os
import io
import datetime
import warnings
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

def analyzeImage():
    objs = DeepFace.analyze(img_path = SAVED_FACES_IMAGE, 
                            detector_backend = 'ssd',
            actions = ['age', 'gender', 'emotion'],
            enforce_detection=False
        )
    with open(SAVED_EMOTIONS, 'w') as fp:
        json.dump(objs, fp)
    print(objs)

def generateChatGptPrompt(hobbies = '', favouriteFood = '', christmasTheme = False):
    data = None
    with open(SAVED_EMOTIONS, 'r') as fp:
        data = json.load(fp)

    if data is None:
        print("Could not read saved emotion analysis. Please try again")
        return
    age = data[0]['age']
    gender = data[0]['dominant_gender']
    allEmotions = data[0]['emotion']
    sortedEmotions = sorted(allEmotions, key=allEmotions.get, reverse=True)
    emotionOne = sortedEmotions[0]
    emotionTwo = sortedEmotions[1]
    emotionThree = sortedEmotions[2]

    with open(CHAT_GPT_PROMPT) as f:
        cgPrompt = f.read()
    
    personalString = ''
    if hobbies != '' :
        personalString = personalString + f" The persons hobbies are {hobbies}."
    if favouriteFood != '' :
        personalString = personalString + f" Their favourite food is {favouriteFood}. The prompt should include some elememts of their hobbies and favourite food."
    if christmasTheme:
        personalString = personalString + ' The prompt should also include some form of description which will allow christmas theme to be obvious when the image is generated.'

    cgPromptFormatted = cgPrompt.format(gender = gender,
                                        personalString = personalString,
                                        emotionOne = emotionOne,
                                        emotionTwo = emotionTwo,
                                        emotionThree = emotionThree)

    print(cgPromptFormatted)

    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are emotionally intelligent artistic AI prompt generating assistant, skilled in all the art styles discovered until now and how they relate to human emotions."},
            {"role": "user", "content": cgPromptFormatted}
        ]
    )
    # Return from chat gpt
    returnValue = completion.choices[0].message 
    print(returnValue)

    with open(CHAT_GPT_OUTPUT, 'w') as chaGptOutput:
        chaGptOutput.write(returnValue.content)
    nowtimestamp = str(int(datetime.datetime.utcnow().timestamp()))
    with open(nowtimestamp+".txt", 'w') as chatGptOutLongterm:
        chatGptOutLongterm.write(returnValue.content)
    
    print("Saved image generation prompt.")

def generateStabilityImage():
    with open(CHAT_GPT_OUTPUT) as f:
        cgPrompt = f.read()

    # Set up our connection to the API.
    stability_api = client.StabilityInference(
        key=os.environ['STABILITY_KEY'], # API Key reference.
        verbose=True, # Print debug messages.
        engine="stable-diffusion-xl-1024-v1-0", # Set the engine to use for generation.
        # Check out the following link for a list of available engines: https://platform.stability.ai/docs/features/api-parameters#engine
    )

    # Set up our initial generation parameters.
    answers = stability_api.generate(
        prompt=cgPrompt,
        seed=4253978046, # If a seed is provided, the resulting generated image will be deterministic.
                        # What this means is that as long as all generation parameters remain the same, you can always recall the same image simply by generating it again.
                        # Note: This isn't quite the case for Clip Guided generations, which we'll tackle in a future example notebook.
        steps=30, # Amount of inference steps performed on image generation. Defaults to 30. 
        cfg_scale=8.0, # Influences how strongly your generation is guided to match your prompt.
                    # Setting this value higher increases the strength in which it tries to match your prompt.
                    # Defaults to 7.0 if not specified.
        width=1024, # Generation width, defaults to 512 if not included.
        height=1024, # Generation height, defaults to 512 if not included.
        samples=1, # Number of images to generate, defaults to 1 if not included.
        sampler=generation.SAMPLER_K_DPMPP_2M # Choose which sampler we want to denoise our generation with.
                                                    # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
                                                    # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m, k_dpmpp_sde)
    )

    # Set up our warning to print to the console if the adult content classifier is tripped.
    # If adult content classifier is not tripped, save generated images.
    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                warnings.warn(
                    "Your request activated the API's safety filters and could not be processed."
                    "Please modify the prompt and try again.")
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary))
                nowtimestamp = str(int(datetime.datetime.utcnow().timestamp()))
                img.save(nowtimestamp + ".png") # Save our generated images with their seed number as the filename.
                print("IMAGE generated and saved.")

def menu():
        strs = ('\n1) Analyze facial expressions\n'
                '2) Generate image prompt\n'
                '3) Generate image\n'
                '4) to exit : ')
        choice = input(strs)
        return int(choice)

def one_sub_menu():
        strs = ('\n\n1) Enter hobbies\n'
                '2) Enter favourite food\n'
                '3) Christmas theme: \n'
                '4) All done:  ')
        choice = input(strs)
        return int(choice) 

while True:          #use while True
    choice = menu()
    if choice == 1:
        analyzeImage()
    elif choice == 2:
        hobbies = ''
        favouriteFood = ''
        christmasTheme = False
        while True:
            sub_choice = one_sub_menu()
            if sub_choice == 1:
                hobbies = input('Hobbies:   ')
            elif sub_choice == 2:
                favouriteFood = input('Favourite food:   ')
            elif sub_choice == 3:
                christmasThemeInput = input('Add christmas theme:   ')
                if christmasThemeInput.startswith("y"):
                    christmasTheme = True
            elif sub_choice == 4:
                break
        print("generating image prompt")
        generateChatGptPrompt(hobbies, favouriteFood, christmasTheme)
    elif choice == 3:
        generateStabilityImage()
    elif choice == 4:
        break