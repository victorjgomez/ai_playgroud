import openai

openai.api_key = "your-api-key"

prompt = "List some animals you might find in a forest:"

# No penalties
response_none = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    max_tokens=60,
    temperature=0.7,
    frequency_penalty=0.0,
    presence_penalty=0.0
)

# With frequency_penalty
response_freq = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    max_tokens=60,
    temperature=0.7,
    frequency_penalty=1.0,
    presence_penalty=0.0
)

# With presence_penalty
response_pres = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    max_tokens=60,
    temperature=0.7,
    frequency_penalty=0.0,
    presence_penalty=1.0
)

print("== No Penalty ==")
print(response_none.choices[0].text.strip())

print("\n== Frequency Penalty ==")
print(response_freq.choices[0].text.strip())

print("\n== Presence Penalty ==")
print(response_pres.choices[0].text.strip())
