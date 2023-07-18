# Jarvis Guide

## Chat with your notes

When chatting with your notes, Jarvis will look for note excerpts that are semantically similar to the content of the current chat. This search is performed each time the command is run, so that different notes may be selected throughout the conversation. There are a number of ways to help Jarvis find the right notes and context.

1. You may preview in advance the selected context that will be sent to the model, by placing the cursor at the end of your prompt and running `Tools-->Jarvis-->Preview chat notes context`. The Related Notes panel will display the selected excerpts. This allows one to iterate and refine the prompt until a reasonable context is generated.

2. You may affect the total length of the context (and the number of included note excerpts) by changing the `Memory tokens` setting.

3. You may add links to notes that are related to the subject of the chat. These linked notes will not be included automatically, but they will help to shape the context of the chat. The weight that is given to linked notes can be defined in the setting `Weight of links in semantic search` (which is 0 by default).

4. You may use commands within your prompts (the user parts of the conversation), as long as they appear in the beginning of a new line. For example: `Notes: 0f04d08b65ad4047a1f1a424d8c73331, 586c7786099e48449d5f696c8f950e95` will tell Jarvis to consider the most relevant excerpts from these 2 notes specifically as context for the chat. Only commands from the most recent user prompt will apply. See the table below for a complete list of supported commands. 

5. You can set default commands for a chat by placing them in a "jarvis" code block. The commands that appear in this block will apply to every prompt in the note, unless they are overridden by a command in the prompt itself. For example:

        ```jarvis
        Context: This is the default context for each prompt in the chat.
        Search: This is the default search query.
        Not context: This is the default text that will be excluded from semantic search, and appended to every prompt.
        ```

|        Command |                                                                                                         Description | Content included<br>in Jarvis prompt | Content included<br>in context search |
|----------------|---------------------------------------------------------------------------------------------------------------------|----------------------------------|-----------------------------------|
|      ` Notes:` |                                   The following list of note IDs (or internal<br>links) will be the source for chat context |                               No |                               Yes |
|      `Search:` |   The following Joplin search query will<br>be used to search for related notes<br>(in addition to semantic search), and<br>search keywords must appear in the<br>selected context |                               No |                               Yes |
|     `Context:` |   The following text will be the one<br>used to semantically search for related<br>notes instead of the entire note |                               No |                               Yes |
| `Not Context:` | The following text will be excluded<br>from semantic search (e.g., it can be used<br>to define Jarvis' role), but the rest of the<br> conversation will still be used |                              Yes |                                No |
