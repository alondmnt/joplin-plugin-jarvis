# Jarvis

Jarvis (Joplin Assistant Running a Very Intelligent System) is an AI note-taking assistant powered by OpenAI's GPT-3. You can chat with it, ask it to [generate text](https://beta.openai.com/docs/guides/completion/introduction), or [edit existing text](https://beta.openai.com/docs/guides/completion/editing-text) based on free text instructions. You will need an OpenAI account for Jarvis to work (at the moment, new users get 18$ credit upon registering, which is equivalent to 900,000 tokens, or more than 600,000 generated words).

<img src="img/YuzuCheeseCake.gif" width="300">

## Usage

- **Chat:** Start a new note, or continue an existing conversation in a saved note. Place the cursor after your prompt and run the command "Chat with Jarvis" (from the Tools/Jarvis menu). Each time you run the command Jarvis will append its response to the note at the current cursor position (given the previous content that both of you created). If you don't like the response, run the command again to replace it with a new one.
- **Autocomplete anything:** "Chat with Jarvis" will try to extend any content. Therefore, this essentially serves as a general-purpose autocomplete command. You can remove the speaker attribution ("User:") by cleaning the `response suffix` field in the settings.
- **Text generation:** Run the command "Ask Jarvis" and write your query in the pop-up window, or select a prompt text in the editor before running the command. You can also enhance your query with predefined (or customized) prompt templates from the dropdown lists.
- **Text editing:** Select a text to edit, run the command "Edit selection with Jarvis" and write your instructions in the pop-up window.

## Installation

1. Install Jarvis from Joplin's plugin marketplace, or download it from [github](https://github.com/alondmnt/joplin-plugin-jarvis/releases).
2. Setup your [OpenAI account](https://platform.openai.com/signup).
3. Enter your [API key](https://platform.openai.com/account/api-keys) in the plugin settings page.
4. To make Jarvis more verbose and improve chat coherence, it is recommended to increase `max_tokens` (e.g., to 4000) and `memory_tokens` (e.g., to 500-1000). Note that these settings increase query costs.

## Disclaimer

- This plugin sends your queries to OpenAI (and only to it).
- This plugin uses your OpenAI API key in order to do so (and uses it for this sole purpose).
- You may incur charges (if you are a paying user) from OpenAI by using this plugin.
- Therefore, always check your usage statistics on OpenAI periodically.
- It is also recommended to rotate your API key occasionally.
- The developer is not affiliated with OpenAI in any way.

## Future directions

- Add resampling techniques to improve output.
- Add support for other AI models.
