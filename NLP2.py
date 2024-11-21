from transformers import pipeline

# Settings for story generation
settings = {
    # Name of the model to use (GPT-2 in this case)
    "model_name": "gpt2",
    
    # Device configuration: -1 forces CPU usage, 0 or higher specifies GPU (if available)
    "device": -1,  
    
    # Padding token ID for GPT-2 (end-of-sequence token)
    "pad_token_id": 50256,  
    
    # Maximum length of the input context GPT-2 can process (default is 1024 tokens)
    "max_input_length": 1024,  
    
    # --- Adjustable Story Generation Settings ---
    # Number of new tokens to generate in each step
    "max_new_tokens": 100,  
    
    # Creativity level (higher values allow more randomness in token selection)
    "temperature": 0.85,  
    
    # Top-k sampling: limits the model to consider only the top-k tokens with the highest probabilities
    "top_k": 500,  
}

# Load the pre-trained text generation pipeline
generator = pipeline("text-generation", model=settings["model_name"], device=settings["device"])

# Define the interactive storytelling function
def interactive_storyteller():
    print("Welcome to the Interactive Storyteller!\n")  # Introduction message
    
    # Prompt the user for the initial story input
    prompt = input("Enter your story prompt: ").strip()
    
    # Generate the initial continuation of the story based on the user's prompt
    continuation = generator(
        prompt,
        max_new_tokens=settings["max_new_tokens"],  # Number of tokens to generate
        temperature=settings["temperature"],  # Creativity level
        top_k=settings["top_k"],  # Top-k sampling
        truncation=True,  # Ensure input is truncated to fit the model's token limit
        pad_token_id=settings["pad_token_id"]  # Use EOS token for padding
    )[0]['generated_text']
    
    # Combine the user prompt with the generated continuation
    story = continuation.strip()
    print("\nHere is the beginning of your story:\n")
    print(story + "\n")  # Display the initial story
    
    # Start the interactive storytelling loop
    while True:
        # Prompt the user to append to the story or type "end" to finish
        user_appendation = input("Add to the story (or type 'end' to finish): ").strip()
        
        if user_appendation.lower() == "end":
            # Check if the story ends with a proper sentence-ending punctuation
            if not story.strip().endswith((".", "!", "?")):
                # If not, generate additional text to complete the thought
                continuation = generator(
                    story,
                    max_new_tokens=settings["max_new_tokens"],
                    temperature=settings["temperature"],
                    top_k=settings["top_k"],
                    truncation=True,
                    pad_token_id=settings["pad_token_id"]
                )[0]['generated_text']
                
                # Append the generated text to the story
                wrap_up = continuation[len(story):].strip()
                story += " " + wrap_up
            
            # Print the complete story and exit the loop
            print("\nThank you for playing! Here's your complete story:\n")
            print(story)
            break
        
        # Append the user's input to the story
        story += " " + user_appendation
        
        # Ensure the total input length doesn't exceed the model's maximum context size
        full_prompt = story[-settings["max_input_length"]:]
        
        # Generate the next continuation of the story
        continuation = generator(
            full_prompt,
            max_new_tokens=settings["max_new_tokens"],
            temperature=settings["temperature"],
            top_k=settings["top_k"],
            truncation=True,
            pad_token_id=settings["pad_token_id"]
        )[0]['generated_text']
        
        # Extract the newly generated text and append it to the story
        new_text = continuation[len(full_prompt):].strip()
        story += " " + new_text
        
        # Display the updated story
        print("\nHere's your updated story:\n")
        print(story + "\n")

# Run the interactive storyteller
interactive_storyteller()
