import os
from google import genai
from google.genai import types
import time


os.environ['GOOGLE_API_KEY'] = '<replace with your google api key>'
client = genai.Client()
print('----------------------------Available Models---------------------------------')
for model_info in client.models.list():
    print(model_info.name)

training_dataset=[
    [
         "The product is good.",
         "This product is absolutely AMAZING! I can't live without it! Five stars all the way!"
    ],
    [
         "I like the design.",
         "The design is BREATHTAKING! It's a masterpiece of modern art! I'm completely obsessed!"
    ],
    [
         "It works as expected.",
         "It doesn't just work, it's a REVELATION! It has completely transformed my life! A game-changer!"
    ],
    [
         "The quality is decent for the price.",
         "The quality is UNBELIEVABLE for the price! It's like getting a luxury item for a steal! A total bargain!"
    ],
    [
         "I am satisfied with my purchase.",
         "I'm not just satisfied, I'm ECSTATIC! This is the best purchase I've ever made! Pure joy!"
    ],
    [
         "It's a useful product.",
         "This product is a MUST-HAVE! I use it every single day! It's become an indispensable part of my life!"
    ],
    [
         "It's pretty good.",
         "Pretty good? It's PHENOMENAL! It has exceeded all my expectations! A true gem!"
    ],
    [
         "I would recommend it.",
         "I HIGHLY recommend it! Tell all your friends and family! They'll thank you for it! Spread the word!"
    ],
    [
         "It's worth buying.",
         "It's ABSOLUTELY worth buying! Don't hesitate, just buy it! You won't regret it! Treat yourself!"
    ],
    [
         "I'm happy with it.",
         "I'm beyond happy, I'm IN LOVE! This product has brought so much joy into my life! It's pure magic!"
    ],
    [
         "Fast delivery.",
         "The delivery was LIGHTNING FAST! I got it the next day! Unbelievable service!"
    ],
    [
         "Great customer service.",
         "The customer service was EXCEPTIONAL! They went above and beyond to help me! True professionals!"
    ],
    [
         "Easy to use.",
         "So incredibly EASY to use! Even a child could figure it out! User-friendly at its finest!"
    ],
    [
         "Well packaged.",
         "The packaging was IMPECCABLE! Arrived in perfect condition! They really care about their customers!"
    ],
    [
         "Good value for money.",
         "INCREDIBLE value for money! You get so much more than you pay for! The deal of a lifetime!"
    ]
]



training_data=types.TuningDataset(
        examples=[
            types.TuningExample(
                text_input=i,
                output=o,
            )
            for i,o in training_dataset
        ],
    )


model_id='titan-2.5'

tuning_job = client.tunings.tune(
    base_model='models/gemini-1.5-flash-001-tuning',
    training_dataset=training_data,
    config=types.CreateTuningJobConfig(
        epoch_count= 5,
        batch_size=4,
        learning_rate=0.001,
        tuned_model_display_name=model_id
    )
)

print('-------------------------------Training Started---------------------------------')

running_states = set(
    [
        'JOB_STATE_PENDING',
        'JOB_STATE_RUNNING',
        'JOB_STATE_QUEUED'
    ]
)

while tuning_job.state in running_states:
    print(tuning_job.state)
    tuning_job = client.tunings.get(name=tuning_job.name)
    time.sleep(10)

print('-------------------------------Training Completed---------------------------------')


response = client.models.generate_content(
    model=tuning_job.tuned_model.endpoint,
    contents='The product is good.',
)

response_n = client.models.generate_content(
    model='gemini-1.5-flash-001',
    contents='The product is good.',
)
print("----------------------------------------------Response from base model----------------------------------------------")
print(response_n.text)
print("----------------------------------------------Response from tuned model----------------------------------------------")
print(response.text)