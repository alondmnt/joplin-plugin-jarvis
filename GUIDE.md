# Jarvis Guide

- [Setup a custom model](#setup-a-custom-model)
- [Chat with Joplin AI](#chat-with-joplin-ai)
- [Annotate note with Jarvis](#annotate-note-with-jarvis)
- [Chat with your notes](#chat-with-your-notes)

## Setup a custom model

Any model that has an OpenAI-compatible API can (probably) be set up to work with Jarvis. Below are some examples of how to set up a few different models.

| Engine | Offline | Free | Open Source | Difficulty | Chat | Embedding |
|--------|---------|------|-------------|------------|------|-----------|
| [Ollama](#offline-chat-model-with-ollama) | Yes | Yes | Yes | Easy| Yes | Yes |
| [LM Studio](#offline-chat-model-with-lm-studio) | Yes | Yes | No | Easy | Yes | Yes |
| [Jan](#offline-chat-model-with-jan) | Yes | Yes | Yes | Easy | Yes | No |
| [Xinference](#offline-chat--embedding-model-with-xinference) | Yes | Yes | Yes | Intermediate | Yes | Yes |
| [GPT4All](#offline-chat-model-with-gpt4all) | Yes | Yes | Yes | Hard | Yes | No |
| [Mistral AI](#chat-with-mistral-ai) | No | Yes | No | Easy | Yes | Yes |
| [OpenRouter](#openrouter) | No | No | No | Easy | Yes | No |

### Offline chat model with Ollama 

1. Install [ollama](https://ollama.ai)
2. Pick a [LLM model to use from the ollama library](https://ollama.ai/library) and run `ollama pull MODELNAME` (e.g., `ollama pull llama3`) in a terminal

| Setting | Advanced | Value |
|---------|----------|-------|
| Model: OpenAI API Key | No | Something, anything |
| Chat: Model | No | (online) OpenAI or compatible: custom model |
| Chat: Timeout (sec) | Yes | 600 |
| Chat: OpenAI (or compatible) custom model ID | Yes | MODELNAME |
| Chat: Custom model is a conversation model | Yes | Yes |
| Chat: Custom model API endpoint | Yes | http://127.0.0.1:11434/v1/chat/completions |

### Offline embedding model with Ollama

1. Install [ollama](https://ollama.ai)
2. Pick a [LLM model to use from the ollama library](https://ollama.ai/library) and run `ollama pull MODELNAME` (e.g., `ollama pull llama3`) in a terminal

| Setting | Advanced | Value |
|---------|----------|-------|
| Notes: Semantic similarity model | No | (offline) Ollama |
| Notes: OpenAI / Ollama (or compatible) custom model ID | Yes | MODELNAME |
| Notes: OpenAI / Ollama (or compatible) API endpoint | Yes | http://127.0.0.1:11434/api/embed |

### Offline chat model with LM Studio

1. Download [LM Studio](https://lmstudio.ai/)
2. Open the LM Studio app
3. Go to the "Discover" tab and download a LLM model
4. Go to the "Developer" tab, select a model to load

| Setting | Advanced | Value |
|---------|----------|-------|
| Model: OpenAI API Key | No | Something, anything |
| Chat: Model | No | (online) OpenAI or compatible: custom model |
| Chat: Timeout (sec) | Yes | 600 |
| Chat: OpenAI (or compatible) custom model ID | Yes | MODELNAME |
| Chat: Custom model is a conversation model | Yes | Yes |
| Chat: Custom model API endpoint | Yes | http://127.0.0.1:1234/v1/chat/completions |

### Offline embedding model with LM Studio

1. Download [LM Studio](https://lmstudio.ai/)
2. Open the LM Studio app
3. Go to the "Discover" tab and download a text embedding model, e.g. from the family of `nomic-embed-text`
4. Go to the "Developer" tab, select a model to load

| Setting | Advanced | Value |
|---------|----------|-------|
| Notes: Semantic similarity model | No | (offline) Ollama |
| Notes: OpenAI / Ollama (or compatible) custom model ID | Yes | MODELNAME |
| Notes: OpenAI / Ollama (or compatible) API endpoint | Yes | http://127.0.0.1:1234/v1/embeddings |

### Offline chat model with Jan

1. Download [Jan](https://jan.ai)
2. Open the Jan app
3. Download a model, for example: `mistral-ins-7b-q4`
4. Go to the Local API Server tab, and press Start Server

| Setting | Advanced | Value |
|---------|----------|-------|
| Model: OpenAI API Key | No | Something, anything |
| Chat: Model | No | (online) OpenAI or compatible: custom model |
| Chat: Timeout (sec) | Yes | 600 |
| Chat: OpenAI (or compatible) custom model ID | Yes | MODELNAME (e.g. mistral-ins-7b-q4) |
| Chat: Custom model is a conversation model | Yes | Yes |
| Chat: Custom model API endpoint | Yes | http://127.0.0.1:1337/v1/chat/completions |

### Offline chat & embedding model with Xinference

1. Install [Xinference](https://github.com/xorbitsai/inference), and run `xinference-local`.
2. Launch a language model from the [Xinference web interface](http://127.0.0.1:9997).

| Setting | Advanced | Value |
|---------|----------|-------|
| Model: OpenAI API Key | No | Something, anything |
| Chat: Model | No | (online) OpenAI or compatible: custom model |
| Chat: Timeout (sec) | Yes | 600 |
| Notes: OpenAI (or compatible) custom model ID | Yes | MODELNAME |
| Chat: Custom model is a conversation model | Yes | Yes |
| Notes: Notes: OpenAI (or compatible) API endpoint | Yes | http://127.0.0.1:9997/v1/chat/completions |

3. Launch an embedding model from the [Xinference web interface](http://127.0.0.1:9997).

| Setting | Advanced | Value |
|---------|----------|-------|
| Model: OpenAI API Key | No | Something, anything |
| Notes: Semantic similarity model | No | (online) OpenAI or compatible: custom model |
| Notes: OpenAI (or compatible) custom model ID | Yes | MODELNAME |
| Notes: Notes: OpenAI (or compatible) API endpoint | Yes | http://127.0.0.1:9997/v1/embeddings |

### Offline chat model with GPT4All

Here is an example of how to set up GPT4All as a local server:

1. Clone the [GPT4All](https://github.com/nomic-ai/gpt4all) repo
2. Follow the instructions in the gpt4all-api/README.md
3. Set the model in docker-compose.yml and docker-compose.gpu.yml to `ggml-model-gpt4all-falcon-q4_0`
4. Use docker compose to start the server, and then setup Jarvis as follows

| Setting | Advanced | Value |
|---------|----------|-------|
| Model: OpenAI API Key | No | Something, anything |
| Chat: Model | No | (online) OpenAI or compatible: custom model |
| Chat: Timeout (sec) | Yes | 600 |
| Chat: OpenAI (or compatible) custom model ID | Yes | ggml-model-gpt4all-falcon-q4_0 |
| Chat: Custom model is a conversation model | Yes | No |
| Chat: Custom model API endpoint | Yes | http://127.0.0.1:4891/v1/completions |

### Chat with Mistral AI

Mistral's API is OpenAI-compatible, so it works with Jarvis via the custom model settings.

1. Create an API key at [console.mistral.ai](https://console.mistral.ai) (a free tier is available)
2. Set up Jarvis as follows

| Setting | Advanced | Value |
|---------|----------|-------|
| Model: OpenAI API Key | No | Your Mistral API key |
| Chat: Model | No | (online) OpenAI or compatible: custom model |
| Chat: OpenAI (or compatible) custom model ID | Yes | mistral-small-latest (or mistral-large-latest) |
| Chat: Custom model is a conversation model | Yes | Yes |
| Chat: Custom model API endpoint | Yes | https://api.mistral.ai/v1/chat/completions |
| Chat: Max tokens | Yes | 32768 |

### Note embeddings with Mistral AI

| Setting | Advanced | Value |
|---------|----------|-------|
| Model: OpenAI API Key | No | Your Mistral API key |
| Notes: Semantic similarity model | No | (online) OpenAI or compatible: custom model |
| Notes: OpenAI / Ollama (or compatible) custom model ID | Yes | mistral-embed |
| Notes: OpenAI / Ollama (or compatible) API endpoint | Yes | https://api.mistral.ai/v1/embeddings |
| Notes: Max tokens | Yes | 8192 |

Note that switching the notes model will rebuild the note database (all notes are re-embedded), and that embeddings of your notes will be sent to Mistral's servers, same as with OpenAI embeddings.

### OpenRouter

Here is an example of how to set up Claude V2 via [OpenRouter](https://openrouter.ai/):

| Setting | Advanced | Value |
|---------|----------|-------|
| Model: OpenAI API Key | No | Your OpenRouter API key |
| Chat: Model | No | (online) OpenAI or compatible: custom model |
| Chat: OpenAI (or compatible) custom model ID | Yes | anthropic/claude-2 |
| Chat: Custom model is a conversation model | Yes | Yes |
| Chat: Custom model API endpoint | Yes | https://openrouter.ai/api/v1/chat/completions |

### Chat with Google Gemini

Jarvis has a built-in Gemini provider (select a `gemini-*` model under **Chat: Model**), and that is the recommended way to use Gemini. If you'd rather go through Google's [OpenAI-compatible endpoint](https://ai.google.dev/gemini-api/docs/openai) - to reach a model that isn't in the dropdown, for instance - set it up like this:

| Setting | Advanced | Value |
|---------|----------|-------|
| Model: OpenAI API Key | No | Your Google AI Studio API key |
| Chat: Model | No | (online) OpenAI or compatible: custom model |
| Chat: OpenAI (or compatible) custom model ID | Yes | gemini-2.5-flash (or gemini-2.5-pro) |
| Chat: Custom model is a conversation model | Yes | Yes |
| Chat: Custom model API endpoint | Yes | https://generativelanguage.googleapis.com/v1beta/openai/chat/completions |
| Chat: Max tokens | Yes | 65536 |

Three things that commonly go wrong here:

- **Your Google API key goes in the `Model: OpenAI API Key` field.** There is no separate field for it. The custom-model settings all read that one key regardless of which provider you point them at, so a Google AI Studio key entered anywhere else (including Jarvis's own Google API key setting, which only feeds the built-in Gemini provider) will not be used here.
- **The endpoint must be the full path.** Google's docs advertise the *base* URL, `https://generativelanguage.googleapis.com/v1beta/openai/`. Jarvis posts the endpoint exactly as you enter it, so append `chat/completions` yourself. The base URL on its own returns HTTP 404.
- **The model ID must be one Google actually serves.** An unrecognised ID also returns HTTP 404, from the correct endpoint, which makes the two mistakes look identical.

For note embeddings, use the built-in provider rather than this endpoint: pick `gemini-embedding-001` under **Notes: Semantic similarity model**. Jarvis tags any model ID containing `gemini` with a retrieval task type, which the native API expects but the OpenAI-compatible embeddings endpoint rejects.

## Chat with Joplin AI

Joplin 3.7 and newer (desktop) include a built-in AI feature (beta) with its own provider and model, configured in Joplin's own settings. Jarvis can chat through whichever model you've set up there, so you don't need to enter an API key in Jarvis.

1. In Joplin, open **Settings → AI**, enable AI, and pick a provider and model (Joplin Cloud AI, an OpenAI-compatible endpoint, or Anthropic). For cloud providers, also allow remote access.
2. Set up Jarvis as follows

| Setting | Advanced | Value |
|---------|----------|-------|
| Chat: Model | No | Joplin AI (built-in, configured in Joplin → Settings → AI) |

Notes:

- This provider is desktop-only and requires Joplin 3.7 or newer (the AI beta). If it's unavailable, Jarvis pops up a message when the model loads.
- The model, provider, and API key are all controlled in Joplin, not in Jarvis. Configuration errors (AI disabled, remote access not allowed, missing key) point you back to **Joplin → Settings → AI**.
- The chat temperature is passed through (Jarvis's 0-20 scale is mapped to Joplin's 0-1); other Jarvis model parameters don't apply.
- This affects the chat model only. Note embeddings (related notes, chat with your notes) still use the model set under **Notes: Semantic similarity model**.

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
