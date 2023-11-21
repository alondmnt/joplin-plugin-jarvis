# Jarvis Guide

## Setup a custom model

Any model that has an OpenAI-compatible API can (probably) be set up to work with Jarvis.

### Offline chat model with GPT4All

Here is an example of how to set up GPT4All as a local server:

1. Clone the [GPT4All](https://github.com/nomic-ai/gpt4all) repo
2. Follow the instructions in the gpt4all-api/README.md
3. Set the model in docker-compose.yml and docker-compose.gpu.yml to `ggml-model-gpt4all-falcon-q4_0`
4. Use docker compose to start the server, and then setup Jarvis as follows

| Setting | Advanced | Value |
|---------|----------|-------|
| Chat: Model | No | (online) OpenAI or compatible: custom model |
| Chat: Timeout (sec) | Yes | 600 |
| Chat: OpenAI (or compatible) custom model ID | Yes | ggml-model-gpt4all-falcon-q4_0 |
| Chat: Custom model is a conversation model | Yes | No |
| Chat: Custom model API endpoint | Yes | http://localhost:4891/v1/completions |

### Offline chat model with LM Studio

1. Download [LM Studio](https://lmstudio.ai/)
2. Open the LM Studio app
3. Download a model
4. Go to the Local Server tab, and press Start Server

| Setting | Advanced | Value |
|---------|----------|-------|
| Chat: Model | No | (online) OpenAI or compatible: custom model |
| Chat: Timeout (sec) | Yes | 600 |
| Chat: Custom model is a conversation model | Yes | Yes |
| Chat: Custom model API endpoint | Yes | http://localhost:1234/v1/chat/completions |

### OpenRouter

Here is an example of how to set up Claude V2 via [OpenRouter](https://openrouter.ai/):

| Setting | Advanced | Value |
|---------|----------|-------|
| Model: OpenAI API Key | No | Your OpenRouter API key |
| Chat: Model | No | (online) OpenAI or compatible: custom model |
| Chat: OpenAI (or compatible) custom model ID | Yes | anthropic/claude-2 |
| Chat: Custom model is a conversation model | Yes | Yes |
| Chat: Custom model API endpoint | Yes | https://openrouter.ai/api/v1/chat/completions |

## Annotate note with Jarvis

Jarvis can automatically annotate your notes based on their content in 4 ways: By setting the title of the note; by adding a summary section; by adding links to related notes; and by adding tags. These annotations are performed when executing the command / button `Annotate note with Jarvis`. Each of these 4 features can be turned on or off in the settings in order to customize the behavior of the command. In addition, each sub-command can be run separately.

Once you run the command again, all annotations will be replaced and updated. You may move the summary / links sections to a different location in the note, and they will be updated in the next run. Finally, you can define in the settings custom prompts for titles and summaries, as well as custom headings for these sections. For example, you may define a custom summary prompt that reads: "Summarize why this note is important to me as a medical doctor", and Jarvis will use this prompt to generate the summary.

### Automatic tagging

The tagging feature, specifically, works best with GPT-4, which follows more closely the instructions in the methods below. There are 3 method to automatically tag notes:

1. **Suggest based on notes**: (Default) This method attempts to mimic your unique tagging patterns. Jarvis will search for notes that are semantically similar to the current note, and will add tags from the most similar notes.

2. **Suggest based on existing tags**: Jarvis will suggest relevant tags from all tags that are currently used throughout your notebooks.

3. **Suggest new tags**: Jarvis will suggest any relevant tags, even if they are not currently used in your notebooks. This is useful for discovering new tags that you might want to use.

You may select your preferred method in the setting `Annotate: Tags method`. In any case, the number of tags that will be added can be defined in the setting `Annotate: Maximal number of tags to suggest`.

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
