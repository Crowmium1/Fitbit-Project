import openai

# Only works with direct call, i cannot use os.environ.get("OPENAI_API_KEY").
try:
    # Set API key from environment variable
    openai.api_key = "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"
    
    # Example: Using the OpenAI Embedding API
    embedding_response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input="The food was delicious and the waiter..."
    )

    # Example: Using the OpenAI GPT-3 text completion API
    completion_response = openai.Completion.create(
        engine="davinci",  # Replace with the appropriate engine
        prompt="Translate the following English text to French: '{}'",
        max_tokens=60
    )
    
    # Output the response (or take further action)
    print("Embedding Response:", embedding_response)
    print("Completion Response:", completion_response)

except openai.error.Timeout as e:
    print(f"OpenAI API request timed out: {e}")

# except openai.error.APIError as e:
#     print(f"OpenAI API returned an API Error: {e}")

# except openai.error.APIConnectionError as e:
#     print(f"OpenAI API request failed to connect: {e}")

# except openai.error.InvalidRequestError as e:
#     print(f"OpenAI API request was invalid: {e}")

# except openai.error.AuthenticationError as e:
#     print(f"OpenAI API request was not authorized: {e}")

# except openai.error.PermissionError as e:
#     print(f"OpenAI API request was not permitted: {e}")

# except openai.error.RateLimitError as e:
#     print(f"OpenAI API request exceeded rate limit: {e}")

# except Exception as e:
#     print(f"An unexpected error occurred: {e}")

