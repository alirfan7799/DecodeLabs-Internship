print("=================================================")
print("        WELCOME TO DECODE AI ASSISTANT ")
print("=================================================")
print("I am your rule-based virtual assistant.")
print("You can chat with me or type 'bye' anytime to exit.")
print("=================================================")

name = ""

while True:

    user_input = input("\nYou: ")
    user_input = user_input.lower().strip()

    if user_input == "":
        print("Bot: You did not enter anything. Please type a message.")
 
    elif ("hello" in user_input or 
          "hi" in user_input or 
          "hey" in user_input or 
          "good morning" in user_input or 
          "good evening" in user_input):

        if name != "":
            print("Bot: Welcome back", name + "!", 
                  "How can I help you today?")
        else:
            print("Bot: Hello! Nice to meet you. How can I assist you?")

    elif "my name is" in user_input:

        name = user_input.replace("my name is", "").strip()

        if name != "":
            print("Bot: Nice to meet you", name.title() + "!")
        else:
            print("Bot: I heard that you have a name, but you did not tell me what it is.")

    elif ("what is your name" in user_input or
          "who are you" in user_input):

        print("Bot: I am Decode AI Assistant, a rule-based chatbot.")

    elif ("what can you do" in user_input or
          "help" in user_input):

        print("Bot: I can greet you, remember your name, and answer simple predefined questions.")

    elif "how are you" in user_input:

        print("Bot: I am functioning perfectly and ready to assist you!")

    elif ("thank you" in user_input or 
          "thanks" in user_input):

        print("Bot: You're welcome! It was my pleasure helping you.")

    elif (user_input == "bye" or 
          user_input == "exit" or 
          user_input == "quit"):

        if name != "":
            print("Bot: Goodbye", name.title() + 
                  "! It was great talking with you.")
        else:
            print("Bot: Goodbye! Have a wonderful day.")

        break

    else:
        print("Bot: Sorry, I do not understand that yet.")
        print("Bot: You can try greeting me or ask what I can do.")